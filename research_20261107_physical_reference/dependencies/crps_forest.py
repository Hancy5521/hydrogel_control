"""Classical 60%-subsampled CRPS forest with exact equal-tree leaf CDFs.

Independent baseline implementation, not copied or patched author code.
The source objective and forest construction are established literature.
"""
from copy import deepcopy
from fractions import Fraction
import math
import random
from crps_tree import CRPSTree,matrix,_vector

class CRPSForest:
 def __init__(self,n_trees=256,min_leaf=3,max_depth=None,subsample_fraction=.6,seed=20261107):
  if type(n_trees) is not int or n_trees<1:raise ValueError('positive integer tree count required')
  if not math.isfinite(subsample_fraction) or not 0<subsample_fraction<=1:raise ValueError('subsample fraction in (0,1] required')
  if type(seed) is not int:raise ValueError('integer seed required')
  CRPSTree(min_leaf,max_depth)
  self.n_trees=n_trees;self.min_leaf=min_leaf;self.max_depth=max_depth;self.fraction=float(subsample_fraction);self.seed=seed;self.trees=[]
 def fit(self,X,y):
  self.X=matrix(X,'X');self.y=_vector(y,'y')
  if len(self.X)!=len(self.y):raise ValueError('X/y row mismatch')
  size=int(self.fraction*len(self.y))
  if size<1:raise ValueError('subsample is empty')
  rng=random.Random(self.seed);self.trees=[];self.subsamples=[]
  for _ in range(self.n_trees):
   indices=rng.sample(range(len(self.y)),size);self.subsamples.append(indices)
   tree=CRPSTree(self.min_leaf,self.max_depth).fit([self.X[i] for i in indices],[self.y[i] for i in indices],row_ids=indices)
   self.trees.append(tree)
  return self
 def laws(self,X_query,unit='native scalar'):
  if len(self.trees)!=self.n_trees:raise ValueError('fit required')
  rows=matrix(X_query,'X_query');result=[]
  for row in rows:
   masses={};leaves=[]
   for tree in self.trees:
    leaf=tree.leaf_for(row);values=leaf['target_values'];leaves.append(leaf['node_id'])
    if not values:raise AssertionError('empty leaf')
    for value in values:masses[value]=masses.get(value,Fraction(0))+Fraction(1,self.n_trees*len(values))
   if sum(masses.values(),Fraction(0))!=1 or any(w<=0 for w in masses.values()):raise AssertionError('exact positive simplex')
   support=sorted(masses)
   result.append({'coordinate':'native','family':'weighted_empirical','support_native':support,
                  'weight_fractions':[str(masses[x]) for x in support],'unit':unit,
                  'leaf_ids_in_tree_order':leaves})
  return result
 def to_state(self):
  if len(self.trees)!=self.n_trees:raise ValueError('fit required')
  return {'kind':'independent_classical_CRPS_subsample_forest','n_trees':self.n_trees,'min_leaf':self.min_leaf,
          'max_depth':self.max_depth,'subsample_fraction':self.fraction,'seed':self.seed,
          'sampling':'Python Random(seed).sample without replacement, fresh draw per tree',
          'aggregation':'equal-tree empirical leaf-CDF mixture; exact Fraction mass',
          'training_X':deepcopy(self.X),'training_y':list(self.y),'subsample_row_indices':deepcopy(self.subsamples),
          'trees':[t.to_state() for t in self.trees],'no_bootstrap_multiplicity_or_OOB_reweighting':True}
