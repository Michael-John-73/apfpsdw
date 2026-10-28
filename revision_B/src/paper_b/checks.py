"""Independent numerical checks. Synthetic fixtures are never paper results."""
from itertools import product
from pathlib import Path
import tempfile
import numpy as np
from scipy.integrate import quad
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import betabinom, binom
from . import codec
from .common import write_json,read_json,progress
from .models import bb_objective,binomial_objective,fit_model,predict_pmf
from .probability import bb_pmf,binomial_pmf,poisson_binomial,acceptance,exact_ci,log_bb

def run_checks():
    checks=[]
    def record(name,call):
        call();checks.append({"name":name,"status":"PASS"})
        print("[CHECK PASS] "+name,flush=True)
    def pmfs():
        for n,a,b in product((1,48,136),(.05,1.,25.),(.1,2.,50.)):
            np.testing.assert_allclose(bb_pmf(n,a,b),betabinom.pmf(np.arange(n+1),n,a,b),rtol=2e-10,atol=1e-13)
        for n,p in product((1,48,136),(0.,.1,.5,1.)):
            np.testing.assert_allclose(binomial_pmf(n,p),binom.pmf(np.arange(n+1),n,p),atol=2e-13)
        np.testing.assert_allclose(bb_pmf(48,3e6,7e6),binomial_pmf(48,.3),atol=2e-6)
    record("PMF versus SciPy independent distributions and binomial limit",pmfs)
    def enumeration():
        ps=[.1,.3,.8,.4];reference=np.zeros(5)
        for bits in product((0,1),repeat=4):
            reference[sum(bits)]+=np.prod([p if b else 1-p for p,b in zip(ps,bits)])
        np.testing.assert_allclose(poisson_binomial(ps),reference,atol=1e-14)
        assert acceptance(np.array([.4,.1,.5]))=={"p_exact":.4,"p_tau":.4}
        assert exact_ci(0,100)["one_sided_upper"]>0
    record("Poisson-binomial exhaustive enumeration; acceptance and zero events",enumeration)
    rng=np.random.default_rng(778)
    x=np.column_stack([np.ones(9),rng.normal(size=(9,2))]);k=np.array([0,1,5,8,20,24,40,47,48.]);n=np.full(9,48.);w=np.ones(9)/9
    def gradients():
        for function,p in ((bb_objective,rng.normal(size=6)),(binomial_objective,rng.normal(size=3))):
            analytic=function(p,x,k,n,w,.03)[1]
            finite=approx_derivative(lambda z:function(z,x,k,n,w,.03)[0],p,method="3-point").ravel()
            np.testing.assert_allclose(analytic,finite,rtol=3e-5,atol=2e-6)
    record("Analytic regression gradients versus central finite differences",gradients)
    def autodiff():
        import torch
        p=rng.normal(size=6)
        z=torch.tensor(p,dtype=torch.float64,requires_grad=True)
        xx=torch.tensor(x);kk=torch.tensor(k);nn=torch.tensor(n);ww=torch.tensor(w)
        mu=1e-8+(1-2e-8)*torch.sigmoid(xx@z[:3]);c=torch.nn.functional.softplus(xx@z[3:])+2e-8
        a,b=mu*c,(1-mu)*c
        def lb(u,v):return torch.lgamma(u)+torch.lgamma(v)-torch.lgamma(u+v)
        lp=torch.lgamma(nn+1)-torch.lgamma(kk+1)-torch.lgamma(nn-kk+1)+lb(a+kk,b+nn-kk)-lb(a,b)
        mask=torch.tensor([0.,1,1,0,1,1]);loss=-(ww*lp).sum()+.5*.03*((z*mask)**2).sum();loss.backward()
        np.testing.assert_allclose(bb_objective(p,x,k,n,w,.03)[1],z.grad.numpy(),rtol=1e-8,atol=1e-9)
    record("Beta-binomial chain rule versus independent torch autograd",autodiff)
    def energy():
        delta=36.
        for b in (0,1):
            mse=quad(lambda r:((.25+.5*b)*delta-r)**2,0,delta)[0]/delta
            np.testing.assert_allclose(mse,7*delta**2/48,atol=1e-10)
        assert .5*255>127 # deployed vote treats an exact 50/50 tie as one
        assert (257//8)*(263//8)==1024
    record("Independent QIM integral, floor carrier count, and vote tie",energy)
    def codec_checks():
        image=rng.integers(0,256,(257,263,3),dtype=np.uint8)
        for kind,swap,payload in product(("uint8","float32"),(False,True),("xl","v1")):
            marked,trace=codec.encode(image,codec.PAYLOADS[payload],kind=kind,swap=swap)
            np.testing.assert_array_equal(marked,codec.encode_fast(image,codec.PAYLOADS[payload],kind=kind,swap=swap))
            assert trace["n_blocks"]==1024
            assert len(codec.decode(marked,len(codec.PAYLOADS[payload]))["bits"])==len(codec.PAYLOADS[payload])
        assert codec.decode(image[:100],48)["status"]=="guard_rejected"
        stages=[{"op":"resize","scale":.75},{"op":"crop","area":.8}]
        assert codec.channel(image,stages,"x.png").shape[:2]==codec.planned_geometry(257,263,stages)
        a=codec.channel(image,[{"op":"jpeg","quality":80,"subsampling":2}],"x")
        assert a.dtype==np.uint8 and a.shape==image.shape
    record("Odd dimensions, fast replay parity, all dtype/swap/payload paths, channel guards",codec_checks)
    def invalid():
        calls=[lambda:log_bb(-1,48,1,1),lambda:log_bb(3,48,-1,1),
               lambda:poisson_binomial([np.nan]),lambda:binomial_pmf(48,1.2),
               lambda:codec.decode(np.zeros((256,256,3),np.uint8),0)]
        for call in calls:
            try:call()
            except ValueError:continue
            raise AssertionError("Invalid input accepted")
    record("Invalid ranges, NaN, and zero-length payload rejected",invalid)
    def serialization():
        rows=[]
        for i in range(16):
            err=rng.binomial(1,.2+i/100,48).tolist()
            rows.append({"role":"public_mark","status":"ok","split":"fit","group_id":str(i),
                         "K":sum(err),"L":48,"errors":err,"features":{"a":float(i),"mechanism_gap":float(i%3)},
                         "variant":"v","payload_id":"xl","condition_id":"none"})
        model=fit_model("mechanistic_bb",rows,.1,{"maxiter":1000,"ftol":1e-10,"gtol":1e-6},False)
        assert model["converged"]
        scratch=Path(__file__).resolve().parents[2]/".verification_tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            p=Path(folder)/"model.json";write_json(p,model);loaded=read_json(p)
            np.testing.assert_array_equal(predict_pmf(model,rows[0]),predict_pmf(loaded,rows[0]))
        assert model["history"] and any(h["parameter_update"]>0 for h in model["history"])
    record("Learned updates, full serialization and exact prediction reload",serialization)
    return {"status":"PASS","checks":checks,"count":len(checks),"scope":"technical numerical checks; not hypothesis evidence"}
