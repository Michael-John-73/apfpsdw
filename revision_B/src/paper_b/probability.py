"""Count distributions and exact finite-sample acceptance probabilities."""
from __future__ import annotations
import numpy as np
from scipy.special import betaln, gammaln, logsumexp, digamma
from scipy.stats import beta

EPS = 1e-8

def log_bb(k, length, alpha, beta_value):
    k, length, a, b = np.broadcast_arrays(np.asarray(k, float), np.asarray(length, float),
                                          np.asarray(alpha, float), np.asarray(beta_value, float))
    if not np.isfinite(k+length).all() or np.any(k < 0) or np.any(k > length) or np.any(k != np.floor(k)) or np.any(length < 1) or np.any(length != np.floor(length)):
        raise ValueError("Invalid integer count or payload length")
    if not np.all(np.isfinite(a+b)) or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Positive finite alpha/beta required")
    return gammaln(length+1)-gammaln(k+1)-gammaln(length-k+1)+betaln(a+k,b+length-k)-betaln(a,b)

def grad_log_bb(k, length, alpha, beta_value):
    a, b = np.asarray(alpha), np.asarray(beta_value)
    common = -digamma(a+b+length)+digamma(a+b)
    return digamma(a+k)-digamma(a)+common, digamma(b+length-k)-digamma(b)+common

def bb_pmf(length, a, b):
    logp = log_bb(np.arange(length+1), length, a, b)
    return np.exp(logp-logsumexp(logp))

def binomial_pmf(length, p):
    if type(length) not in (int,np.int64,np.int32) or length<1:raise ValueError("Positive integer length required")
    if not 0 <= p <= 1:
        raise ValueError("Invalid probability")
    if p in (0, 1):
        out=np.zeros(length+1)
        out[0 if p==0 else length]=1
        return out
    k=np.arange(length+1)
    logp=gammaln(length+1)-gammaln(k+1)-gammaln(length-k+1)+k*np.log(p)+(length-k)*np.log1p(-p)
    return np.exp(logp-logsumexp(logp))

def poisson_binomial(probabilities):
    p=np.asarray(probabilities,float)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Finite Bernoulli probabilities required")
    pmf=np.ones(1)
    for q in p:
        next_p=np.zeros(len(pmf)+1)
        next_p[:-1]+=pmf*(1-q)
        next_p[1:]+=pmf*q
        pmf=next_p
    return pmf/pmf.sum()

def temperature_pmf(pmf, temperature):
    if not .25 <= temperature <= 4:
        raise ValueError("Temperature outside the prespecified numerical range")
    logp=np.log(np.maximum(np.asarray(pmf,float),1e-300))/temperature
    return np.exp(logp-logsumexp(logp))

def acceptance(pmf, tau=.125):
    p=np.asarray(pmf,float)
    if p.ndim!=1 or np.any(p<0) or not np.isclose(p.sum(),1,atol=1e-9):
        raise ValueError("Normalized PMF required")
    length=len(p)-1
    return {"p_exact":float(p[0]),"p_tau":float(p[:int(np.floor(tau*length))+1].sum())}

def exact_ci(k,n,confidence=.95):
    if n<=0 or not 0<=k<=n:
        raise ValueError("Invalid event counts")
    tail=(1-confidence)/2
    return {"k":int(k),"n":int(n),"rate":k/n,
            "low":0. if k==0 else float(beta.ppf(tail,k,n-k+1)),
            "high":1. if k==n else float(beta.ppf(1-tail,k+1,n-k)),
            "one_sided_upper":1. if k==n else float(beta.ppf(confidence,k+1,n-k)),
            "confidence":confidence}

def paired_interval(differences, seed=20261021, reps=5000, confidence=.975):
    d=np.asarray(differences,float)
    if d.ndim!=1 or len(d)<2 or not np.isfinite(d).all() or reps<100:
        raise ValueError("At least two finite independent group values required")
    rng=np.random.default_rng(seed)
    means=np.empty(reps)
    for start in range(0,reps,100):
        n=min(100,reps-start)
        means[start:start+n]=d[rng.integers(0,len(d),size=(n,len(d)))].mean(axis=1)
    lo,hi=np.quantile(means,[(1-confidence)/2,1-(1-confidence)/2])
    return {"estimate":float(d.mean()),"low":float(lo),"high":float(hi),"n_groups":len(d),
            "confidence":confidence,"reps":reps}

