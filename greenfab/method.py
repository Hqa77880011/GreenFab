import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;k=c.get('outcomes',4);self.net=nn.Sequential(nn.Linear(c.get('input_shape',[8])[-1],32),nn.SiLU(),nn.Linear(32,3*k))
    def forward(self,b):
        mean,lo,hi=self.net(b['x'].float()).chunk(3,-1);return {'prediction':mean,'lower':mean-F.softplus(lo),'upper':mean+F.softplus(hi)}
    def objective(self,b,p):
        y=b['y'].float();loss=F.mse_loss(p['prediction'],y)
        for q,key in [(.05,'lower'),(.95,'upper')]:
            residual=y-p[key];loss=loss+torch.maximum(q*residual,(q-1)*residual).mean()
        return loss


def demo(seed,n,c):
    rng=np.random.default_rng(seed);x=rng.uniform(0,1,(n,8)).astype('float32');return {'x':x,'y':np.stack([x[:,0],x[:,1]**2,x[:,2]+x[:,3],x[:,4]*x[:,5]],1).astype('float32'),'group':np.arange(n)}


def optimize(problem,variant='full'):
    from scipy.optimize import linprog
    outcomes=np.asarray(problem['objectives']);count,scenarios,k=outcomes.shape;prob=np.asarray(problem['probabilities']);distance=np.asarray(problem['distance']);robust=np.zeros((count,k))
    equality=np.zeros((scenarios,scenarios*scenarios))
    for source in range(scenarios):equality[source,source*scenarios:(source+1)*scenarios]=1
    for a in range(count):
        for j in range(k):
            if variant in {'baseline','no_robustness'}:robust[a,j]=prob@outcomes[a,:,j];continue
            result=linprog(-np.tile(outcomes[a,:,j],scenarios),A_ub=distance.reshape(1,-1),b_ub=[problem['radius']],A_eq=equality,b_eq=prob,bounds=(0,None),method='highs')
            if not result.success:raise ValueError('Wasserstein scenario transport is infeasible: '+result.message)
            robust[a,j]=-result.fun
    constraints=[]
    if variant!='no_preferences':
        for preferred,other in problem.get('preferences',[]):constraints.append(robust[preferred]-robust[other])
    admissible=problem.get('feasible',[True]*count);regret=[]
    for a in range(count):
        if not admissible[a]:regret.append(None);continue
        worst=0.
        for rival in range(count):
            if not admissible[rival]:continue
            result=linprog(-(robust[a]-robust[rival]),A_ub=np.array(constraints) if constraints else None,b_ub=np.zeros(len(constraints)) if constraints else None,A_eq=np.ones((1,k)),b_eq=[1],bounds=(0,1),method='highs')
            if not result.success:raise ValueError('Preference answers are mutually inconsistent')
            worst=max(worst,-result.fun)
        regret.append(float(worst))
    allowed=[i for i,r in enumerate(regret) if r is not None]
    if not allowed:raise ValueError('No physically feasible allocation candidate')
    selected=min(allowed,key=lambda i:regret[i]);return {'selected':selected,'robust_objectives':robust.tolist(),'minimax_regret':regret,'preference_constraints':[row.tolist() for row in constraints]}


def execute(args):
    import json
    from pathlib import Path
    result=optimize(json.loads(Path(args.input).read_text()),args.variant);out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'allocation.json').write_text(json.dumps(result,indent=2))
