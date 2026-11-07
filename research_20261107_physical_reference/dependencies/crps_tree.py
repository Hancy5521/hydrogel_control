"""Classical CRPS-CART baseline, independently implemented from its objective.

See Duchemin & Obozinski, arXiv:2502.05157v3. This is baseline infrastructure,
not a new learning algorithm and not a modification of their Python source.
Prefix empirical entropy is half the pairwise absolute energy divided by n².
One bucket per distinct target value makes target ties independent of order.
Only the Python standard library is required.
"""
import math
from copy import deepcopy

EPS=2.220446049250313e-16

def _vector(values,name):
 result=[float(v) for v in values]
 if any(not math.isfinite(v) for v in result):raise ValueError(name+' must be finite')
 return result

def matrix(values,name):
 result=[_vector(row,name) for row in values]
 if not result or not result[0] or any(len(row)!=len(result[0]) for row in result):
  raise ValueError(name+' must be a nonempty rectangular matrix')
 return result

class _CountsAndSums:
 def __init__(self,size):self.count=[0]*(size+1);self.total=[0.]*(size+1)
 def add(self,index,value):
  i=index+1
  while i<len(self.count):
   self.count[i]+=1;self.total[i]+=value;i+=i&-i
 def prefix(self,end):
  count=0;terms=[];i=end
  while i:
   count+=self.count[i];terms.append(self.total[i]);i-=i&-i
  return count,math.fsum(terms)

def prefix_entropy(targets):
 """H_k=Σ_{i<j≤k}|y_i−y_j|/k², for k=0..N; O(N log N)."""
 values=_vector(targets,'targets')
 if not values:return [0.]
 origin=min(values);centered=[v-origin for v in values]
 if any(not math.isfinite(v) for v in centered):raise ArithmeticError('target span overflows')
 support=sorted(set(centered));bucket={v:i for i,v in enumerate(support)}
 sums=_CountsAndSums(len(support));pair_energy=0.;result=[0.]
 for n,value in enumerate(centered):
  rank=bucket[value];less_n,less_sum=sums.prefix(rank);le_n,le_sum=sums.prefix(rank+1)
  all_n,all_sum=sums.prefix(len(support))
  increment=math.fsum([value*less_n,-less_sum,all_sum,-le_sum,-value*(all_n-le_n)])
  if not math.isfinite(increment) or increment<0:
   raise ArithmeticError('nonfinite or negative absolute-distance increment; no clipping')
  pair_energy=math.fsum([pair_energy,increment]);risk=pair_energy/((n+1)*(n+1))
  if not math.isfinite(risk):raise ArithmeticError('nonfinite empirical CRPS entropy')
  result.append(risk);sums.add(rank,value)
 return result

def _threshold(left,right):
 """Any representable t with left≤t<right yields the intended ≤ partition."""
 if not left<right:raise ValueError('distinct ordered feature values required')
 value=left/2+right/2
 if value>=right:value=math.nextafter(right,left)
 if not left<=value<right or not math.isfinite(value):raise ArithmeticError('unrepresentable split')
 return value

def _tolerance(a,b):return 64*EPS*max(abs(a),abs(b))

class CRPSTree:
 def __init__(self,min_leaf=3,max_depth=None):
  if type(min_leaf) is not int or min_leaf<1:raise ValueError('positive integer min_leaf required')
  if max_depth is not None and (type(max_depth) is not int or max_depth<0):raise ValueError('nonnegative depth or None required')
  self.min_leaf=min_leaf;self.max_depth=max_depth;self.root=None
 def fit(self,X,y,row_ids=None):
  self.X=matrix(X,'X');self.y=_vector(y,'y');n=len(self.X)
  if len(self.y)!=n:raise ValueError('X/y row mismatch')
  self.row_ids=list(range(n)) if row_ids is None else list(row_ids)
  if len(self.row_ids)!=n or len(set(self.row_ids))!=n or any(type(i) is not int for i in self.row_ids):
   raise ValueError('unique integer source-row indices required')
  self.features=len(self.X[0]);self.root=self._grow(list(range(n)),0,'root');return self
 def _grow(self,rows,depth,node_id):
  targets=[self.y[i] for i in rows];parent=prefix_entropy(targets)[-1]
  node={'node_id':node_id,'rows':[self.row_ids[i] for i in rows],'n':len(rows),'entropy_native':parent}
  if len(rows)<2*self.min_leaf or (self.max_depth is not None and depth>=self.max_depth) or parent==0:
   return dict(node,leaf=True,target_values=targets)
  best=None;n=len(rows)
  for feature in range(self.features):
   ordered=sorted(rows,key=lambda i:(self.X[i][feature],i));ys=[self.y[i] for i in ordered]
   left=prefix_entropy(ys);right=prefix_entropy(reversed(ys))
   for count in range(self.min_leaf,n-self.min_leaf+1):
    a=self.X[ordered[count-1]][feature];b=self.X[ordered[count]][feature]
    if not a<b:continue
    threshold=_threshold(a,b);risk=math.fsum([count*left[count],(n-count)*right[n-count]])/n
    if not parent-risk>_tolerance(parent,risk):continue
    key=(risk,feature,threshold)
    if best is None or key<best[0]:best=(key,count)
  if best is None:return dict(node,leaf=True,target_values=targets)
  (risk,feature,threshold),count=best
  left_rows=[i for i in rows if self.X[i][feature]<=threshold]
  right_rows=[i for i in rows if self.X[i][feature]>threshold]
  if len(left_rows)!=count or min(len(left_rows),len(right_rows))<self.min_leaf:
   raise AssertionError('saved threshold differs from evaluated partition')
  return dict(node,leaf=False,feature=feature,threshold=threshold,child_weighted_entropy_native=risk,
              left=self._grow(left_rows,depth+1,node_id+'L'),right=self._grow(right_rows,depth+1,node_id+'R'))
 def leaf_for(self,row):
  if self.root is None:raise ValueError('fit required')
  values=_vector(row,'query')
  if len(values)!=self.features:raise ValueError('query feature count differs')
  node=self.root
  while not node['leaf']:node=node['left'] if values[node['feature']]<=node['threshold'] else node['right']
  return node
 def to_state(self):
  if self.root is None:raise ValueError('fit required')
  return {'kind':'independent_classical_CRPS_CART','min_leaf':self.min_leaf,'max_depth':self.max_depth,
          'feature_count':self.features,'all_features_searched':True,'training_X':deepcopy(self.X),
          'training_y':list(self.y),'source_row_indices':list(self.row_ids),'root':deepcopy(self.root),
          'routing_train_and_inference':'<= left, > right','strict_gain_roundoff_tolerance':'64*eps*max(abs(parent),abs(child))'}
