"""Two CPU-only numerical paths for declared E1 semantics.

Path 1 executes a logical block in request/layer/token order using scalar dot
products. Path 2 interprets the actual compiled COMPUTE/COPY/SHARE/EVICT actions,
using vectorized NumPy and physical per-domain buffers. They share only fixed
input generation and IDs, not intermediate answers or operator implementations.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
from experiments.dac_validation_e1_oracle import tid,kv,weight,need


NUMERICAL_CONTRACT = dict(
    dense="F32 RMSNorm(epsilon=1e-5), Q/K/V linear, adjacent-pair normal RoPE(base=10000), causal GQA, O linear+residual, RMSNorm/SwiGLU FFN+residual; trace-supplied input hidden at each round.",
    moe="Explicit test arithmetic on opaque .6 ports: three HxH packed matrices projection tanh(x@(W0+W1+W2).T/sqrt(3)); causal mean of all prior/current packed projection rows plus current projection plus 0.25*input; KV is the packed projection row; model trace supplies top-1 router ID; packed gate/up/down SwiGLU per selected row; combine=attention+selected expert output. W_router is an explicit residency/readiness input but route IDs remain model-owned trace values, not a learned numerical router.",
    moe_scope="Synthetic operator contract solely for scheduling/dataflow preservation; physical=None and opaque projection are NOT a production numerical MoE implementation.",
    two_paths="scalar logical block reference versus vectorized actual compiled-action interpreter with physical buffers/copy completion; every declared layer/round hidden and KV port compared.",
    dtype="F32 materialized ports; F64 internal arithmetic in both CPU test paths; no native kernel timing or production precision claim",
    input_seed=20261024,atol=5e-5,rtol=5e-5)


def values(identity,size,seed,*,gamma=False,weight_tensor=False):
    local=int(hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()[:16],16)
    rng=np.random.default_rng(local)
    z=rng.uniform(-0.5,0.5,size)
    if gamma:z=1+0.4*z
    elif weight_tensor:z=z/math.sqrt(max(1,int(math.sqrt(size))))
    return z.astype(np.float32)


def weight_values(s,l,name):
    h,i,d=s["hidden"],s["intermediate"],s["head_dim"]
    shapes={"W_input_norm":(h,),"W_q":(h,h),"W_k":(d,h),"W_v":(d,h),
        "W_output":(h,h),"W_post_attention_norm":(h,),"W_gate":(i,h),
        "W_up":(i,h),"W_down":(h,i),"W_proj":(3,h,h),"W_router":(3,h),
        **{f"E{e}":(3*h*i,) for e in range(3)}}
    shape=shapes[name];size=math.prod(shape)
    return values(weight(l,name),size,s["seed"],gamma=name in {"W_input_norm","W_post_attention_norm"},weight_tensor=True).reshape(shape)


def trace_hidden(s,r,step):
    n=s["q"] if step==0 else 1
    return values(f"trace:request{r}:step{step}",n*s["hidden"],s["seed"]).reshape(n,s["hidden"])


def logical_reference(s):
    """Mathematical block oracle; never reads operations, compiler graph or trace."""
    h,i,d=s["hidden"],s["intermediate"],s["head_dim"]
    out={};history={}
    def cast(x):return np.asarray(x,dtype=np.float32)
    def dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a,b))
    def scalar_mm(x,w):return np.asarray([[dot(row,col) for col in w] for row in x],dtype=np.float64)
    def rms(x,gamma):
        return np.asarray([[float(v)/math.sqrt(math.fsum(float(z)**2 for z in row)/len(row)+1e-5)*float(gamma[j]) for j,v in enumerate(row)] for row in x])
    def scalar_ffn(x,gate,up,down):
        a=scalar_mm(x,gate);b=scalar_mm(x,up)
        activated=np.asarray([[v/(1+math.exp(-v))*float(b[t,j]) for j,v in enumerate(row)] for t,row in enumerate(a)])
        return scalar_mm(activated,down)
    for r in range(s["requests"]):
        for step in range(s["decode_steps"]+1):
            x=trace_hidden(s,r,step).astype(np.float64);past=0 if step==0 else s["q"]+step-1;n=len(x)
            for l in range(s["layers"]):
                if s["kind"]=="dense":
                    normalized=rms(x,weight_values(s,l,"W_input_norm"))
                    qkv=[]
                    for role,heads in (("q",2),("k",1),("v",1)):
                        z=scalar_mm(normalized,weight_values(s,l,"W_"+role)).reshape(n,heads,d)
                        if role!="v":
                            for t in range(n):
                                for head in range(heads):
                                    for pair in range(d//2):
                                        angle=(past+t)/10000**(2*pair/d);a,b=z[t,head,2*pair:2*pair+2]
                                        z[t,head,2*pair]=a*math.cos(angle)-b*math.sin(angle)
                                        z[t,head,2*pair+1]=a*math.sin(angle)+b*math.cos(angle)
                        z=cast(z);qkv.append(z.astype(np.float64))
                        out[tid(r,step,l,"proj."+role)]=z.transpose(1,0,2).ravel()
                    q,k,v=qkv
                    old=history.get((r,l),[])
                    keys=[row[:d] for row in old]+[k[t,0] for t in range(n)]
                    vals=[row[d:] for row in old]+[v[t,0] for t in range(n)]
                    context=np.zeros((n,h))
                    for t in range(n):
                        for head in range(2):
                            scores=[dot(q[t,head],keys[j])/math.sqrt(d) for j in range(past+t+1)]
                            largest=max(scores);ex=[math.exp(z-largest) for z in scores];total=math.fsum(ex)
                            for col in range(d):context[t,head*d+col]=math.fsum(ex[j]/total*float(vals[j][col]) for j in range(len(ex)))
                    attention=cast(scalar_mm(context,weight_values(s,l,"W_output"))+x)
                    out[tid(r,step,l,"attn")]=attention.ravel()
                    updated=[]
                    for t in range(n):
                        block=cast(np.r_[k[t,0],v[t,0]]);out[kv(r,l,past+t)]=block;updated.append(block.astype(np.float64))
                    history[r,l]=[*old,*updated]
                    norm=rms(attention.astype(np.float64),weight_values(s,l,"W_post_attention_norm"))
                    ffn=cast(scalar_ffn(norm,weight_values(s,l,"W_gate"),weight_values(s,l,"W_up"),weight_values(s,l,"W_down"))+attention)
                    out[tid(r,step,l,"ffn")]=ffn.ravel();x=ffn.astype(np.float64)
                else:
                    packed=weight_values(s,l,"W_proj");w=packed[0].astype(np.float64)+packed[1]+packed[2]
                    linear=scalar_mm(x,w)/math.sqrt(3)
                    proj=cast([[math.tanh(float(v)) for v in row] for row in linear])
                    out[tid(r,step,l,"proj")]=proj.ravel()
                    old=history.get((r,l),[]);allrows=[*old,*list(proj)]
                    attn=np.zeros_like(proj)
                    for t in range(n):
                        for c in range(h):attn[t,c]=float(proj[t,c])+math.fsum(float(row[c]) for row in allrows[:past+t+1])/(past+t+1)+0.25*float(x[t,c])
                        out[kv(r,l,past+t)]=proj[t].copy()
                    history[r,l]=allrows;out[tid(r,step,l,"attn")]=attn.ravel()
                    combined=attn.copy()
                    for t in range(n):
                        p=past+t;e=p%2 if p<s["q"] else s["future"]
                        pack=weight_values(s,l,"E"+str(e));gate=pack[:i*h].reshape(i,h);up=pack[i*h:2*i*h].reshape(i,h);down=pack[2*i*h:].reshape(h,i)
                        combined[t]+=cast(scalar_ffn(attn[t:t+1],gate,up,down))[0]
                    combined=cast(combined);out[tid(r,step,l,"combine")]=combined.ravel();x=combined.astype(np.float64)
    return out


def interpreted_actions(s,expected,graph,events):
    """Execute *observed* operands/actions; no expected numerical values are read."""
    tensor={t["id"]:t for t in graph["workload"]["tensors"]}
    ops={o["id"]:o for o in graph["workload"]["operations"]}
    context=expected["contexts"];buffers={};produced={};pending=[];copychecks=0
    h,i,d=s["hidden"],s["intermediate"],s["head_dim"]
    def matrix(z):return np.asarray(z,dtype=np.float64).reshape(-1,np.asarray(z).shape[-1])
    def mm(x,w):return np.asarray(x,dtype=np.float64)@np.asarray(w,dtype=np.float64).T
    def rms(x,g):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)*g
    def cast(z):return np.asarray(z,dtype=np.float32).ravel()
    for ti,t in tensor.items():
        if t["role"]!="weight":continue
        parts=ti.split(":");l=int(parts[1][5:]);name=parts[2];z=weight_values(s,l,name)
        if ":ffn:" in ti:
            shard=parts[-1];start,end=(0,i//4) if shard=="host" else (i//4,i)
            z=z[:,start:end] if name=="W_down" else z[start:end]
        z=cast(z)
        for loc in t.get("initial_locations",[]):buffers[ti,loc]=z.copy()
    def shaped(ti,loc):
        need((ti,loc) in buffers,"numeric input missing in actual device buffer: "+ti)
        return buffers[ti,loc].astype(np.float64).reshape(tuple(reversed(tensor[ti]["ne"])))
    def flat(ti,loc):return shaped(ti,loc).ravel()
    def attention_parts(op,c,loc,q=None,k=None,v=None,prob=None):
        past,n=c["past"],c["n"]
        keys=np.zeros((past+n,d));vals=np.zeros_like(keys)
        history=[ti for ti in op["inputs"] if ti.startswith("kv:")]
        for ti in history:
            p=int(ti.rsplit(":",1)[1]);row=flat(ti,loc);keys[p]=row[:d];vals[p]=row[d:]
        if k is not None:keys[past:]=k[0,0]
        if v is not None:vals[past:]=v[0,0]
        if prob is None:
            scores=np.stack([q[0,head]@keys.T/math.sqrt(d) for head in range(2)])
            return scores
        return np.stack([prob[0,head]@vals for head in range(2)]).transpose(1,0,2).reshape(n,h)
    def finish(t):
        nonlocal pending,copychecks
        rest=[]
        for task in pending:
            if task[0]>t:rest.append(task);continue
            _,kind,updates=task
            for key,z in updates.items():buffers[key]=z
            if kind=="COMPUTE":
                for (ti,_),z in updates.items():produced[ti]=z.copy()
            elif kind=="COPY":copychecks+=1
        pending=rest
    for e in events:
        finish(e["t_ns"]);kind=e["kind"];target=e.get("target_id")
        if kind=="ADVANCE":continue
        if kind=="EVICT":buffers.pop((target,e["location"]));continue
        if kind in {"COPY","SHARE"}:
            source=(target,e["src"]);dest=(target,e["dst"])
            need(source in buffers,"numeric copy before source completion")
            z=buffers[source].copy() if kind=="COPY" else buffers[source]
            if kind=="SHARE":buffers[dest]=z
            else:pending.append((e["end_ns"],kind,{dest:z}))
            continue
        need(kind=="COMPUTE","unknown numeric action")
        op=ops[target];c=context[target];r,step,l,n,past=(c[k] for k in ("r","step","l","n","past"))
        loc="dram" if e["device"]=="cpu" else "vram";k=op["semantic_op"]
        inp=[shaped(ti,loc) for ti in op["inputs"]];outs=[op["output"],*op.get("extra_outputs",[])];z={}
        if k=="state_update":
            # Consume predecessor buffers above, but no sampling/embedding:
            # next hidden and positions are explicit external trace values.
            z[outs[0]]=trace_hidden(s,r,step)
            for ti in outs[1:]:z[ti]=np.arange(past,past+n,dtype=np.int32)
        elif s["kind"]=="dense":
            if k=="projection":
                x=inp[-2].reshape(n,h);normalized=rms(x,inp[0].ravel())
                for index,heads in enumerate((2,1,1)):
                    a=mm(normalized,inp[1+index].reshape(-1,h)).reshape(n,heads,d)
                    if index<2:
                        angle=np.arange(past,past+n)[:,None]/(10000**(np.arange(0,d,2)/d))[None,:]
                        even=a[:,:,0::2].copy();odd=a[:,:,1::2].copy()
                        a[:,:,0::2]=even*np.cos(angle)[:,None,:]-odd*np.sin(angle)[:,None,:]
                        a[:,:,1::2]=even*np.sin(angle)[:,None,:]+odd*np.cos(angle)[:,None,:]
                    z[outs[index]]=a.transpose(1,0,2)
            elif k in {"attention","attention_qk"}:
                if k=="attention":q_,k_,v_=inp[1:4]
                else:q_,k_=inp[:2]
                scores=attention_parts(op,c,loc,q=q_,k=k_)
                if k=="attention_qk":z[outs[0]]=scores
                else:
                    for t in range(n):scores[:,t,past+t+1:]=-np.inf
                    p=np.exp(scores-scores.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
                    ctx=attention_parts(op,c,loc,k=k_,v=v_,prob=p[None])
                    z[outs[0]]=mm(ctx,inp[0].reshape(h,h))+inp[4].reshape(n,h)
                    for t,ti in enumerate(outs[1:]):z[ti]=np.r_[k_[0,0,t],v_[0,0,t]]
            elif k=="attention_softmax":
                a=inp[0].copy()
                for t in range(n):a[:,:,t,past+t+1:]=-np.inf
                p=np.exp(a-a.max(axis=-1,keepdims=True));z[outs[0]]=p/p.sum(axis=-1,keepdims=True)
            elif k=="attention_pv":
                p,k_,v_=inp[:3];z[outs[0]]=attention_parts(op,c,loc,k=k_,v=v_,prob=p)
                for t,ti in enumerate(outs[1:]):z[ti]=np.r_[k_[0,0,t],v_[0,0,t]]
            elif k=="attention_output":z[outs[0]]=mm(inp[0].reshape(n,h),inp[1].reshape(h,h))+inp[2].reshape(n,h)
            elif k=="dense_ffn":
                x=inp[4].reshape(n,h);a=rms(x,inp[0].ravel());gate=mm(a,inp[1].reshape(i,h));up=mm(a,inp[2].reshape(i,h))
                z[outs[0]]=mm(gate/(1+np.exp(-gate))*up,inp[3].reshape(h,i))+x
            elif k=="ffn_norm":z[outs[0]]=rms(inp[1].reshape(n,h),inp[0].ravel())
            elif k in {"ffn_gate","ffn_up","ffn_down"}:
                start,end=op["op_params"];width=end-start
                w=inp[0].reshape((h,i) if k=="ffn_down" and s["pattern"]=="A12" else (i,h) if s["pattern"]=="A12" else (h,width) if k=="ffn_down" else (width,h))
                if s["pattern"]=="A12":w=w[:,start:end] if k=="ffn_down" else w[start:end]
                z[outs[0]]=mm(inp[1].reshape(n,w.shape[1]),w)
            elif k=="ffn_swiglu":z[outs[0]]=inp[0]/(1+np.exp(-inp[0]))*inp[1]
            elif k in {"ffn_sum","ffn_residual"}:z[outs[0]]=sum(inp)
            else:raise AssertionError("unsupported Dense numeric op "+k)
        else:
            if k=="projection":
                w=inp[0].ravel().reshape(3,h,h).sum(axis=0)
                z[outs[0]]=np.tanh(mm(inp[1].reshape(n,h),w)/math.sqrt(3))
            elif k=="attention":
                proj=inp[0].reshape(n,h);x=inp[1].reshape(n,h)
                old=np.asarray([flat(ti,loc) for ti in op["inputs"] if ti.startswith("kv:")])
                rows=np.concatenate((old.reshape(past,h),proj),axis=0)
                mean=np.cumsum(rows,axis=0)/(np.arange(past+n)+1)[:,None]
                z[outs[0]]=proj+mean[past:]+0.25*x
                for t,ti in enumerate(outs[1:]):z[ti]=proj[t]
            elif k=="router":z[outs[0]]=np.asarray([p%2 if p<s["q"] else s["future"] for p in range(past,past+n)],dtype=np.int32)
            elif k=="expert_ffn":
                route=inp[2].ravel().astype(int);selected=np.flatnonzero(route==c["expert"])
                x=inp[1].reshape(n,h)[selected];pack=inp[0].ravel()
                gate=mm(x,pack[:i*h].reshape(i,h));up=mm(x,pack[i*h:2*i*h].reshape(i,h))
                z[outs[0]]=mm(gate/(1+np.exp(-gate))*up,pack[2*i*h:].reshape(h,i))
            elif k=="combine":
                a=inp[0].reshape(n,h).copy();route=inp[1].ravel().astype(int)
                for ti,data in zip(op["inputs"][2:],inp[2:]):
                    expert=int(ti.rsplit(".",1)[1]);selected=np.flatnonzero(route==expert)
                    a[selected]+=data.reshape(len(selected),h)
                z[outs[0]]=a
            else:raise AssertionError("unsupported synthetic MoE numeric op "+k)
        need(set(z)==set(outs),"numeric action output port mismatch")
        updates={}
        for ti,a in z.items():
            array=np.asarray(a,dtype=np.int32 if tensor[ti]["dtype"]=="GGML_TYPE_I32" else np.float32).ravel()
            need(len(array)==math.prod(tensor[ti]["ne"]),"numeric output shape mismatch")
            need(np.all(np.isfinite(array)),"nonfinite numeric output")
            updates[ti,loc]=array
        pending.append((e["end_ns"],"COMPUTE",updates))
    finish(max(e["end_ns"] for e in events));need(not pending,"numeric pending outputs")
    return produced,copychecks


def check_numerical(s,expected,graph,events):
    reference=logical_reference(s)
    interpreted,copies=interpreted_actions(s,expected,graph,events)
    worst=0.;ports=[]
    for ti,want in reference.items():
        need(ti in interpreted,"numeric missing declared logical port "+ti)
        got=interpreted[ti];want=np.asarray(want,dtype=np.float32)
        need(got.shape==want.shape,"numeric port shape differs")
        error=float(np.max(np.abs(got.astype(np.float64)-want)))
        worst=max(worst,error)
        need(np.allclose(got,want,atol=s["atol"],rtol=s["rtol"]),f"numeric mismatch {ti}, max_abs={error}")
        ports.append(dict(id=ti,elements=len(got),max_absolute_error=error))
    return dict(status="pass",contract="complete_declared_dense" if s["kind"]=="dense" else "synthetic_opaque_moe",
        checked_logical_ports=len(ports),checked_elements=sum(p["elements"] for p in ports),
        interpreted_materialized_ports=len(interpreted),physically_replayed_copy_values=copies,
        max_absolute_error=worst,atol=s["atol"],rtol=s["rtol"],ports=ports,
        outputs=dict(reference={ti:np.asarray(z).tolist() for ti,z in reference.items()},
            interpreted={ti:interpreted[ti].tolist() for ti in reference}))
