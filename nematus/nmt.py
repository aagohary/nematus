'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import json
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time

from subprocess import Popen

from collections import OrderedDict

from data_iterator import TextIterator
from util import load_dict
from alignment_util import *

profile = False


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    """
    if use_noise:
      drop half of state_before at random
    else:
      downweight all of state_before by 0.5
    (in expectation, they are the same)
    input: any Theano variable
    output: the input variable, dropped out
    """
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


# dropout that will be re-used at different time steps
def shared_dropout_layer(shape, use_noise, trng, value):
    """
    value is dropout rate
    output: a variable that can be multiplied by some state to get a dropped out version of that state.
    i.e., state * shared_dropout_layer(...) <==> dropout_layer(state, ...)
    """
    proj = tensor.switch(
        use_noise,
        trng.binomial(shape, p=value, n=1,
                                     dtype='float32'),
        theano.shared(numpy.float32(value)))
    return proj

# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          }
    # TODO: why is this strings that get eval'd rather than just
    # function pointers?


def get_layer(name):
    fns = layers[name]
    # see above TODO
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    """
    generate a random ndim*ndim matrix whose columns
    are orthogonal (and maybe rows too)
    (eigenvectors of a gaussian random matrix)
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    """
    initialize a random weight matrix of size:
      nin*nout, or nin*nin if nout is not specified
    if ortho=True and matrix is square, then generate orthogonal matrix
    otherwise, generate gaussian random matrix with std=scale
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=30000):
    """
    input:
      seqs_x: list of sentences (a sentence is a list of word ids), input
      seqs_y: list of sentences (a sentence is a list of word ids), output
      maxlen: throw out anything where either input or output is >= maxlen
      
    output:
      x, x_mask, y, y_mask which are all LEN*N_SAMPLES matrices
      x,y are the word ids, and the masks==0. indicates padding
    """
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:   # throw out sentences that are too long
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1   # length of longest input + 1
    maxlen_y = numpy.max(lengths_y) + 1   # length of longest output + 1 <-- presumably +1 is for bigram connections on output layer

    # x, y are matrices of size LEN*N_SAMPLES of word ids
    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    # x_mask, y_mask are matrices of size LEN*N_SAMPLES == 1. if
    #   this is an observed word, and == 0. if padding
    #   padding on the right
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    """
    initialize parameters for a feed-foward network
    includes weights W (nin*nout) -- if None, read as dim_proj from options
    and bias b (1*nout)

    W is initialized by norm_weight, scale=0.01, ortho as argument
    b is initialized to zero
    """
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    """
    computes feed-forward activations
    state_below = input
    tparams = theano parameters
    activ = activation
    output = result of activ(state_below*W + b)
    """
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    """
    initialize parameters for GRU
    reminder about GRU:
        input is h_prev and x
        r     = sigmoid( W[0] x + U[0] h_prev )    <-- reset
        u     = sigmoid( W[1] x + U[1] h_prev )    <-- gating
        h_tmp = tanh ( Wx x + r . (Ux h_prev) )
        h     = u.h_prev + (1-u).h_tmp          (. = component-wise product)
        everything has bias also
      so shapes:
        W[0]: nin x dim
        W[1]: nin x dim <--- these are packed into one matrix W
        Wx  : nin x dim
        U[0]: dim x dim
        U[1]: dim x dim <--- these are packed into one matrix U
        Ux  : dim x dim
    initialization for all biases is zero
    for square matrices is orthogonal
    and non-square matrices is gaussian random with default variance
    """
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              emb_dropout=None,
              rec_dropout=None,
              **kwargs):
    """
    state_below = input to GRU, matrix of size (#words)*(#seqs)*(inp-dim)
    mask is (#words)*(#seq)

    emb_dropout: 
    rec_dropout: dropout of hidden state
    
    state_below_ = input * W + b
    state_belowx = input * Wx + bx
    
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim): 
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below*emb_dropout[0], tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below*emb_dropout[1], tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux, rec_dropout):
        # m_  = mask for this time step
        # x_  = state_below_ for this time step
        # xx_ = state_belowx for this time step
        # h_  = previous hidden state
        # preact = dropout(h_) * U + x_, called pre-sigmoid "r" in GRU notes
        preact = tensor.dot(h_*rec_dropout[0], U) + x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim)) # split in half
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim)) # split in half

        # compute the hidden state proposal
        # preactx = (dropout(h_) * Ux) . r + xx_, pre-tanh "h_tmp"
        preactx = tensor.dot(h_*rec_dropout[1], Ux) * r + xx_

        # hidden state proposal, previously called h_tmp
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        # if mask==0, then h = h_ (h_prev) <-- during padding, just copy
        # if mask==1, then h = what we just computed

        # return new hidden state
        return h

    # prepare scan arguments
    # seqs: the sequences that we'll process, TODO figure out what these are
    # init_state: initial hidden state, which is zero vectors
    # _step: the function that scan will reduce over
    # shared_vars: GRU needs U variables because Wx and Wx x will be precomputed
    # rec_dropout: ???
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')],
                   rec_dropout]

    # do reduction over seqs
    # will return rval=all of the hidden states
    # and updates is ignored
    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
# TODO: come back to this later
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    """
    initialize parameters for conditional GRU.
    we run a typical GRU on h_prev and x to get "prime" variables:
        input is h_prev and x
        r'     = sigmoid( W[0] x + U[0] h_prev )    <-- reset
        u'     = sigmoid( W[1] x + U[1] h_prev )    <-- gating
        h_tmp' = tanh ( Wx x + r' . (Ux h_prev) )
        h'     = u'.h_prev + (1-u').h_tmp'          (. = component-wise product)
    then attention-context c is computed as (called "context")
        e      = U_att * tanh( W_comb_at s' + W_comb_att encodercontext ) ) + c_tt  <-- there is an error in pdf
        alpha  = softmax( e )
        ctx_   = E_alpha[ encoder-context ]   # encoder-context is called cc_
    then a second GRU, with h_prev = h' and x=c
        r     = sigmoid( Wc[0] ctx_ + U_nl[0] h' )    <-- reset
        u     = sigmoid( Wc[1] ctx_ + U_nl[1] h' )    <-- gating
        h_tmp (=_s_j) = tanh( Wx ctx_ + r . (Ux h'))   <-- the same, except c instead of x
        h = u.h' + (1-u).h_tmp, u is called z in pdf
    """
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl
    params[_p(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')

    Ux_nl = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None, emb_dropout=None,
                   rec_dropout=None, ctx_dropout=None,
                   **kwargs):
    """
    input:
      state_below = x
      mask as before
      context = sequence of forward/backward encodings of input
      rest is the same as gru_layer()
    """
    
    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    # pctx_ = dropout(context) * Wc_att + b_att
    # this is actually Wa h in the pdf
    pctx_ = tensor.dot(context*ctx_dropout[0], tparams[_p(prefix, 'Wc_att')]) +\
        tparams[_p(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    # state_belowx = dropout(x) * Wx + bx
    # state_below_ = dropout(x) * W  + b
    state_belowx = tensor.dot(state_below*emb_dropout[0], tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below*emb_dropout[1], tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_, rec_dropout, ctx_dropout,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):
        # m_ mask, x_=x*W+b, xx_=x*Wx+bx, h_=encoder_context
        # ctx_=
        # 
        
        preact1 = tensor.nnet.sigmoid( tensor.dot(h_*rec_dropout[0], U) + x_ )
        r1 = _slice(preact1, 0, dim)  # this is r'
        u1 = _slice(preact1, 1, dim)  # this is u'
        preactx1 = tensor.dot(h_*rec_dropout[1], Ux) * r1 + xx_  # this is pre-tanh of h_tmp'
        h1 = tensor.tanh(preactx1)  # this is h_tmp'

        h1 = u1 * h_ + (1. - u1) * h1  # this is h'
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # at this point, we've run the first GRU to get h',
        # it's parameters were U and Ux

        # attention
        pstate_ = tensor.dot(h1*rec_dropout[2], W_comb_att)
        pctx__  = tensor.tanh(pctx_ + pstate_[None, :, :])   # originally +xc_)
        alpha = tensor.dot(pctx__*ctx_dropout[1], U_att)+c_tt
        # here, alpha should be = e
        # alpha = Uatt * tanh( [W_comb_att encoder_context] + h' * W_comb_att ) + c_tt
        #  where [...] = pctx_
        
        # alpha = [tanh(pctx_ + {h' * W_comb_att}]*U_att + c_tt
        # so pctx_ should be U_att s'
        # there seems to be an error: GRU pdf says W_comb_att*context, but here we have W_comb_att*h'; pdf is wrong, should be Wa*s'_j not Wa*h
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        # softmax the alphas
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context
        # ctx_ is expectation (under alpha) of encoder_context

        preact2 = tensor.nnet.sigmoid(tensor.dot(h1*rec_dropout[3], U_nl)+b_nl + \
                                      tensor.dot(ctx_*ctx_dropout[2], Wc))
        r2 = _slice(preact2, 0, dim) # this is r
        u2 = _slice(preact2, 1, dim) # this is u

        preactx2 = (tensor.dot(h1*rec_dropout[4], Ux_nl)+bx_nl) * r2 + \
                   tensor.dot(ctx_*ctx_dropout[3], Wcx)
        h2 = tensor.tanh(preactx2) # this is h_tmp

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1
        # this is h
        
        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    # pass pre-multipled x in
    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context, rec_dropout, ctx_dropout] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0])],
                                    non_sequences=[pctx_, context, rec_dropout, ctx_dropout]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    # Wemb maps input word id -> dim_word embeddings
    # Wemb_dec maps output word id -> dim_word embeddings
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])
    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    # encoder: bidirectional RNN
    # build two GRUs, 'encoder' and 'encoder_r' for forward/backward encodings
    # options['encoder'] is either 'gru' or 'gru_cond'
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    params = get_layer(options['encoder'])[0](options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    # dimension of context is 2dim because we have forward+backward
    # encodings at each state
    ctxdim = 2 * options['dim']

    # init_state, init_cell
    # connection from (the final states of the input RNNs) -> initial state of the output RNN
    # call the first part "input representation"
    # fully connected
    params = get_layer('ff')[0](options, params, prefix='ff_state',
                                nin=ctxdim, nout=options['dim'])
    
    # decoder, GRU over output embeddings,
    # every output also gets to look at the "input representation"
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              dimctx=ctxdim)
    # readout
    # ff_logit_lstm: initial state of GRU (dim) -> output embedding
    # ff_logit_prev: output embedding -> output embedding
    # ff_logit_ctx : attention context -> output embedding
    # ff_logit     : output embedding -> output vocabulary
    params = get_layer('ff')[0](options, params, prefix='ff_logit_lstm',
                                nin=options['dim'],
                                nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit_ctx',
                                nin=ctxdim,
                                nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])

    return params


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    # .tag.test_value is some debugging thing (see http://deeplearning.net/software/theano/tutorial/debug_faq.html)
    # x, x_mask, y, y_mask: theano variables, see above
    x = tensor.matrix('x', dtype='int64')
    x.tag.test_value = (numpy.random.rand(5, 10)*100).astype('int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    x_mask.tag.test_value = numpy.ones(shape=(5, 10)).astype('float32')
    y = tensor.matrix('y', dtype='int64')
    y.tag.test_value = (numpy.random.rand(8, 10)*100).astype('int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')
    y_mask.tag.test_value = numpy.ones(shape=(8, 10)).astype('float32')

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[::-1]
    xr_mask = x_mask[::-1]

    # timesteps = length of sequence
    # n_samples = number of sequences
    n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[1]

    if options['use_dropout']:
        # make dropout layers for everything we possibly can
        # this is intended for training time
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_source = 1-options['dropout_source']
        retain_probability_target = 1-options['dropout_target']
        rec_dropout = shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden)
        rec_dropout_r = shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden)
        rec_dropout_d = shared_dropout_layer((5, n_samples, options['dim']), use_noise, trng, retain_probability_hidden)
        emb_dropout = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb)
        emb_dropout_r = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb)
        emb_dropout_d = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb)
        ctx_dropout_d = shared_dropout_layer((4, n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden)
        source_dropout = shared_dropout_layer((n_timesteps, n_samples, 1), use_noise, trng, retain_probability_source)
        target_dropout = shared_dropout_layer((n_timesteps_trg, n_samples, 1), use_noise, trng, retain_probability_target)
        source_dropout = tensor.tile(source_dropout, (1,1,options['dim_word']))
        target_dropout = tensor.tile(target_dropout, (1,1,options['dim_word']))
    else:
        # variables of same size that say don't drop out
        # this is intended for test time
        rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
        rec_dropout_d = theano.shared(numpy.array([1.]*5, dtype='float32'))
        emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([1.]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([1.]*4, dtype='float32'))

    # word embedding for forward rnn (source)
    # take input (x.flatten), map to embeddings (first line), then reshape appropriately
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    if options['use_dropout']:
        emb *= source_dropout

    # run forward RNN on input
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask,
                                            emb_dropout=emb_dropout, 
                                            rec_dropout=rec_dropout)
    
    # word embedding for backward rnn (source)
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    if options['use_dropout']:
        embr *= source_dropout[::-1]

    # run backward rnn on input
    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask,
                                             emb_dropout=emb_dropout_r,
                                             rec_dropout=rec_dropout_r)

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    if options['use_dropout']:
        ctx_mean *= shared_dropout_layer((n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden)

    # initial decoder state, ctx_mean -> feed forward layer, tanh'd
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    # at this point, emb is output embeddings. PERIOD.
    
    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    # at this point, emb_shifted = [0 emb[:-1]]
    # so original emb must be padded on right by something
    emb = emb_shifted
    # we're now ignoring the original?
    # TODO: figure out what is going on here.... hypothesis: comments and code are inconsistent

    if options['use_dropout']:
        emb *= target_dropout

    # decoder - pass through the decoder conditional gru with attention
    # gru_cond returns:
    #    hidden states
    #    attention context vectors
    #    alpha = argmax of context vectors???
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=y_mask,
                                            context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state,
                                            emb_dropout=emb_dropout_d,
                                            ctx_dropout=ctx_dropout_d,
                                            rec_dropout=rec_dropout_d)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    if options['use_dropout']:
        proj_h *= shared_dropout_layer((n_samples, options['dim']), use_noise, trng, retain_probability_hidden)
        emb *= shared_dropout_layer((n_samples, options['dim_word']), use_noise, trng, retain_probability_emb)
        ctxs *= shared_dropout_layer((n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden)

    # weights (alignment matrix) #####LIUCAN: this is where the attention vector is.
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    # probabilities look like:
    #   probs = softmax( logit )
    #   logit = linear( dropout[ tanh{ lstm + prev + ctx } ] )
    #   lstm  = linear( initial state of decoder GRU )
    #   prev  = linear( embedding of the current word )  <-- TODO: check this
    #   ctx   = linear( weighted averages of context from attention )
    logit_lstm = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    if options['use_dropout']:
        logit *= shared_dropout_layer((n_samples, options['dim_word']), use_noise, trng, retain_probability_hidden)

    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    
    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    # compute negative log likelihood of each true word under
    # predicted softmax probability distribution
    #
    # this is the cost that's getting fed into sgd etc...
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)
    # ignore costs of words that were masked out, and add them up
    
    #print "Print out in build_model()"
    #print opt_ret
    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost


def build_sampler(tparams, options, use_noise, trng, return_alignment=False):
    """
    build a sampler, to sample an output at test time

    return pair of Theano functions:
      f_init: take x (input seq) -> the initial state for the decoder, and context vector
      f_next: take y (output prefix, context vector, previous decoder state)
                  -> (probability distribution over words,
                      sample from that distribution,
                      next decoder state)
      (what you need to run the decoder step-by-step)
    """
    x = tensor.matrix('x', dtype='int64')
    xr = x[::-1]
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source), forward and backward
    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    embr = tparams['Wemb'][xr.flatten()]
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])

    if options['use_dropout']:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_source = 1-options['dropout_source']
        retain_probability_target = 1-options['dropout_target']
        rec_dropout = theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))
        rec_dropout_d = theano.shared(numpy.array([retain_probability_hidden]*5, dtype='float32'))
        emb_dropout = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([retain_probability_hidden]*4, dtype='float32'))
        source_dropout = theano.shared(numpy.float32(retain_probability_source))
        target_dropout = theano.shared(numpy.float32(retain_probability_target))
        emb *= source_dropout
        embr *= source_dropout
    else:
        rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
        rec_dropout_d = theano.shared(numpy.array([1.]*5, dtype='float32'))
        emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([1.]*2, dtype='float32'))
        ctx_dropout_d = theano.shared(numpy.array([1.]*4, dtype='float32'))

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix='encoder', emb_dropout=emb_dropout, rec_dropout=rec_dropout)


    projr = get_layer(options['encoder'])[1](tparams, embr, options,
                                             prefix='encoder_r', emb_dropout=emb_dropout_r, rec_dropout=rec_dropout_r)

    # concatenate forward and backward rnn hidden states
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)

    if options['use_dropout']:
        ctx_mean *= retain_probability_hidden

    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print >>sys.stderr, 'Building f_init...',
    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print >>sys.stderr, 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    if options['use_dropout']:
        emb *= target_dropout

    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx,
                                            one_step=True,
                                            init_state=init_state,
                                            emb_dropout=emb_dropout_d,
                                            ctx_dropout=ctx_dropout_d,
                                            rec_dropout=rec_dropout_d)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]

    # alignment matrix (attention model)
    dec_alphas = proj[2]

    if options['use_dropout']:
        next_state_up = next_state * retain_probability_hidden
        emb *= retain_probability_emb
        ctxs *= retain_probability_hidden
    else:
        next_state_up = next_state

    logit_lstm = get_layer('ff')[1](tparams, next_state_up, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    if options['use_dropout']:
        logit *= retain_probability_hidden

    logit = get_layer('ff')[1](tparams, logit, options,
                              prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print >>sys.stderr, 'Building f_next..',
    inps = [y, ctx, init_state]
    outs = [next_probs, next_sample, next_state]

    if return_alignment:
        outs.append(dec_alphas)

    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print >>sys.stderr, 'Done'

    return f_init, f_next


def gen_next_word(f_init, f_nexts, x, y, return_alignment=False):
    """
    Ahmed
    returns a distribution (sum if ensemble) over the next word in a given output prefix
    x: source sentence
    y: output prefix
    TODO: return_alignment if needed
    """        
    num_models = len(f_init)
    dec_state = [None]*num_models
    ctx = [None]*num_models
    next_p = [None]*num_models
    
    # get initial state of decoder and encoder context
    for i in xrange(num_models):
        dec_state[i], ctx[i] = f_init[i](x)
    
    #??? seems that there is a bug with updating next_w in gen_sample     
    #updating the decoder state with the given prefix. 
    for w in [-1]+y: # -1 is a bos indicator
        w = w * numpy.ones((1,)).astype('int64')
        for i in xrange(num_models):            
            inps = [w, ctx[i], dec_state[i]] 
            next_p[i], next_w_tmp, dec_state[i] = f_next[i](*inps)

    return sum(next_p)[0] 


def gen_sample(f_init, f_next, x, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False, return_alignment=False, suppress_unk=False):
    """
    generate an output.
    some ways of doing this:
      (1) stochastic=True argmax=False --> sample output sequence GREEDILY by sampling from p(y_t|x)
      (2) stochastic=True argmax=True  --> generate output sequence GREEDILY by choosing argmax p(y_t|x)
      (3) stochastic=False             --> beam search with beam-size = k
    [note: if stochastic=False and k=1, is this the same as (2)]
     
    generate sample, either with stochastic sampling or beam search. Note that,
    this function iteratively calls f_init and f_next functions.
    """               
    
    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []                   # sequence of predicted words
    sample_score = []             # corresponding score(s)
    sample_word_probs = []        # beam search TODO?
    alignment = []
    if stochastic:
        sample_score = 0          # in non-beam search, just one score

    live_k = 1                    # num of elements on beam
    dead_k = 0                    # num of completed hypotheses

    hyp_samples = [[]] * live_k
    word_probs = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    if return_alignment:
        hyp_alignment = [[] for _ in xrange(live_k)]

    # for ensemble decoding, we keep track of states and probability distribution
    # for each model in the ensemble
    num_models = len(f_init)
    next_state = [None]*num_models   # state of conditional language model (changes over time)
    ctx0 = [None]*num_models         # context vector from input rnns (constant over time)
    next_p = [None]*num_models       # distribution over words
    dec_alphas = [None]*num_models   # alignments
    # get initial state of decoder rnn and encoder context
    for i in xrange(num_models):
        next_state[i], ctx0[i] = f_init[i](x)       # get initial state and context for each model
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    # x is a sequence of word ids followed by 0 (eos id)
    for ii in xrange(maxlen):
        for i in xrange(num_models):
            ctx = numpy.tile(ctx0[i], [live_k, 1])  # make live_k copies of context
            inps = [next_w, ctx, next_state[i]]     # input to LM = word, context and state for each item on beam
            ret = f_next[i](*inps)                  # let rnn take a step on that input
            # dimension of dec_alpha (k-beam-size, number-of-input-hidden-units)
            next_p[i], next_w_tmp, next_state[i] = ret[0], ret[1], ret[2]
            if return_alignment:
                dec_alphas[i] = ret[3]

            if suppress_unk:
                next_p[i][:,1] = -numpy.inf
        if stochastic:   # aka "not beam search"
            if argmax:
                # [0] below is because there's only one element in the "beam"
                nw = sum(next_p)[0].argmax()   # next_p is vector of probabilities, sum over models, take argmax -- this is basically a mixture model over the models
            else:
                nw = next_w_tmp[0]             # this is a sample from next_p
            sample.append(nw)                  # we got a word, append it
            sample_score += next_p[0][0, nw]   # update score
            # TODO there's a bug here: we need to update something like:
            # next_w = nw * numpy.ones((1,)).astype('int64')
            if nw == 0:                        # eos
                break
        else:  # beam search mode
            # next_p[i,j,k] = p(word k | model i, previous hypothesis j) <-- indices might be slightly wrong
            probs       = sum(next_p)/num_models     # average word probability over models
            probs_flat  = probs.flatten()            # p(word k | prev hyp j) avg'd over models i

            cand_scores = hyp_scores[:, None] - sum(numpy.log(next_p))  # [j,k] = path score for hyp k -> word j
            cand_flat   = cand_scores.flatten()
            # argpartition is like argsort, but you only care about the top (selection algorithm)
            ranks_flat  = cand_flat.argpartition(k-dead_k-1)[:(k-dead_k)] # only care about k-dead_k top elements

            #averaging the attention weights accross models
            if return_alignment:
                mean_alignment = sum(dec_alphas)/num_models

            voc_size = next_p[0].shape[1]
            # index of each k-best hypothesis, mapping from flattened version to non-flattened version
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            # now construct subsequence beam state
            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_word_probs = []
            new_hyp_states = []
            if return_alignment:
                # holds the history of attention weights for each time step for each of the surviving hypothesis
                # dimensions (live_k * target_words * source_hidden_units]
                # at each time step we append the attention weights corresponding to the current target word
                new_hyp_alignment = [[] for _ in xrange(k-dead_k)]

            # idx is enumeration
            # ti -> index of k-best hypothesis ... aka which beam element are we extending
            # wi -> index of word to extend it with
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])  # extend ti'th beam with word wi
                new_word_probs.append(word_probs[ti] + [probs_flat[ranks_flat[idx]].tolist()]) # corresponding probability of this word under each model
                new_hyp_scores[idx] = copy.copy(costs[idx])   # shallow copy
                new_hyp_states.append([copy.copy(next_state[i][ti]) for i in xrange(num_models)])
                if return_alignment:
                    # get history of attention weights for the current hypothesis
                    new_hyp_alignment[idx] = copy.copy(hyp_alignment[ti])
                    # extend the history with current attention weights
                    new_hyp_alignment[idx].append(mean_alignment[ti])


            # check the finished samples (aka those that have predicted eos=0)
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            word_probs = []
            if return_alignment:
                hyp_alignment = []

            # sample and sample_score hold the k-best translations and their scores
            # why is this logic here and not in the prev loop?
            for idx in xrange(len(new_hyp_samples)):    # enumerate each of the new beam elements
                if new_hyp_samples[idx][-1] == 0:       # if it predicted eos
                    sample.append(new_hyp_samples[idx]) # then this is a final output, incr dead_k
                    sample_score.append(new_hyp_scores[idx])
                    sample_word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        alignment.append(new_hyp_alignment[idx])
                    dead_k += 1
                else:   # predicted a real word
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        hyp_alignment.append(new_hyp_alignment[idx])
            hyp_scores = numpy.array(hyp_scores)

            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            # get the state for the decoder at the next time step
            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = [numpy.array(state) for state in zip(*hyp_states)]

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
                sample_word_probs.append(word_probs[idx])
                if return_alignment:
                    alignment.append(hyp_alignment[idx])

    if not return_alignment:
        alignment = [None for i in range(len(sample))]

    return sample, sample_score, sample_word_probs, alignment


# calculate the log probablities on a given corpus using translation model.
# essentially map f_log_probs over iterator, producing some json output along the way.
# used to compute scores on dev data for eg early stopping
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True, normalize=False, alignweights=False):
    probs = []
    n_done = 0

    alignments_json = []

    for x, y in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'])

        ### in optional save weights mode.
        if alignweights:
            pprobs, attention = f_log_probs(x, x_mask, y, y_mask)
            for jdata in get_alignments(attention, x_mask, y_mask):
                alignments_json.append(jdata)
        else:
            pprobs = f_log_probs(x, x_mask, y, y_mask)

        # normalize scores according to output length
        if normalize:
            lengths = numpy.array([numpy.count_nonzero(s) for s in y_mask.T])
            pprobs /= lengths

        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

        print "predicted probs length = ",len(probs)
        print probs

    return numpy.array(probs), alignments_json


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


"""Ahmed: restartable
init_accumulators is a dict of:
    running_up2_init: dicts of param key and its running update. 
    running_grads2_init: dicts of param key and its running grad.
init to zero if None
"""
def adadelta(lr, tparams, grads, inp, cost, init_accumulators=None):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()] #always set to zero -- grads are just copied later
    
    if init_accumulators is None:
        running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                     name='%s_rup2' % k)
                       for k, p in tparams.iteritems()] 

        running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                        name='%s_rgrad2' % k)
                          for k, p in tparams.iteritems()] 

    else:
        running_up2 = [theano.shared(init_accumulators['running_up2']['%s_rup2' % k],
                                name='%s_rup2' % k)
                            for k in tparams.keys()] 

        running_grads2 = [theano.shared(init_accumulators['running_grads2']['%s_rgrad2' % k],
                                name='%s_rgrad2' % k)
                            for k in tparams.keys()] 
        

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    accumulators = {} # dict of lists of shared variables for storage purposes
    accumulators['running_up2'] = running_up2
    accumulators['running_grads2'] = running_grads2    
    return f_grad_shared, f_update, accumulators


""" Ahmed: Replaced with a restartable implementation (see above)
def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()] 

    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]  

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update
"""

def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up,
                                    profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup,
                                    profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update

#Ahmed
def get_accumlators_values(accumlators):
    store = {}
    for k,v in accumlators.iteritems():
        d = {}
        for vi in v:
            d[vi.name]=vi.get_value()
        store[k] = d
    return store


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          encoder='gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words_src=100000,  # source vocabulary size
          n_words=100000,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
          valid_datasets=['../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok'],
          dictionaries=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
          use_dropout=False,
          dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
          dropout_hidden=0.5, # dropout for hidden layers (0: no dropout)
          dropout_source=0, # dropout source words (0: no dropout)
          dropout_target=0, # dropout target words (0: no dropout)
          reload_=False,  
          init_accumulators_path=None,
          model_reload_path=None,
          overwrite=False,
          external_validation_script=None,
          shuffle_each_epoch=True,
          finetune=False):


    # Model options
    model_options = locals().copy()

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        worddicts[ii] = load_dict(dd)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'Reloading model options'
        try:
            with open('%s.json' % saveto, 'rb') as f:
                model_options = json.load(f)
        except:
            with open('%s.pkl' % saveto, 'rb') as f:
                model_options = pkl.load(f)

    # reload accumlators
    init_accumulators = None
    if reload_ and init_accumulators_path is not None and os.path.exists(init_accumulators_path):
        print 'Reloading optimizer accumlators'
        init_accumulators = pkl.load(open(init_accumulators_path, 'rb'))



    print 'Loading data'
    train = TextIterator(datasets[0], datasets[1],
                         dictionaries[0], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen,
                         shuffle_each_epoch=shuffle_each_epoch)
    valid = TextIterator(valid_datasets[0], valid_datasets[1],
                         dictionaries[0], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen)

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(model_reload_path):
        print 'Reloading model parameters'
        params = load_params(model_reload_path, params)

    tparams = init_tparams(params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)

    inps = [x, x_mask, y, y_mask]

    print 'Building sampler'
    f_init, f_next = build_sampler(tparams, model_options, use_noise, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    # allow finetuning with fixed embeddings
    if finetune:
        updated_params = OrderedDict([(key,value) for (key,value) in tparams.iteritems() if key not in ['Wemb', 'Wemb_dec']])
    else:
        updated_params = tparams

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(updated_params))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
 
    print 'Building optimizers...', ##Ahmed: because of accumulators -- will crash if optimizer =/= adadelta
    f_grad_shared, f_update, accumulators = eval(optimizer)(lr, updated_params, grads, inps, cost, init_accumulators)
    print 'Done'

    print 'Optimization'

    best_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        history_errs = list(rmodel['history_errs'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    for eidx in xrange(max_epochs): #Restart: eidx
        n_samples = 0

        for x, y in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen,
                                                n_words_src=n_words_src,
                                                n_words=n_words)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y, y_mask)

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving the best model...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
                json.dump(model_options, open('%s.json' % saveto, 'wb'), indent=2)
                print 'Done'

                # save with uidx
                if not overwrite:
                    print 'Saving the model at iteration {}...'.format(uidx),
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    numpy.savez(saveto_uidx, history_errs=history_errs,
                                uidx=uidx, **unzip(tparams))
                    print 'Saving the accumulators at iteration {}...'.format(uidx),
                    
                    with open(saveto_uidx+'accumulators', 'wb') as outfile:
                        pkl.dump(get_accumlators_values(accumulators), outfile, protocol=pkl.HIGHEST_PROTOCOL)

                    print 'Done'


            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[1])):
                    stochastic = True
                    sample, score, sample_word_probs, alignment = gen_sample([f_init], [f_next],
                                               x[:, jj][:, None],
                                               trng=trng, k=1,
                                               maxlen=30,
                                               stochastic=stochastic,
                                               argmax=False,
                                               suppress_unk=False)
                    print 'Source ', jj, ': ',
                    for vv in x[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[0]:
                            print worddicts_r[0][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Truth ', jj, ' : ',
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            print worddicts_r[1][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[1]:
                            print worddicts_r[1][vv],
                        else:
                            print 'UNK',
                    print

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print 'Valid ', valid_err

                if external_validation_script:
                    print "Calling external validation script"
                    print 'Saving  model...',
                    params = unzip(tparams)
                    numpy.savez(saveto +'.dev', history_errs=history_errs, uidx=uidx, **params)
                    json.dump(model_options, open('%s.dev.npz.json' % saveto, 'wb'), indent=2)
                    print 'Done'
                    p = Popen([external_validation_script])

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    
    valid_err, alignment = pred_probs(f_log_probs, prepare_data,
                           model_options, valid)
    valid_err = valid_err.mean()

    print 'Valid ', valid_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **params)

    return valid_err


if __name__ == '__main__':
    pass
