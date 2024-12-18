---
layout: post
title: "Free-Wilson edge-cases"
date: 2024-11-10
---
I've always been a fan of Free-Wilson Analysis. It just gets a lot of things _right_: it's easy-to-use, it helps you understand SAR better and it generate all potentially interesting combinations of your compounds that you might have missed. What's not to love? 


Pat Walters has written up a wonderful post on Free-Wilson Analysis over on [Practical Cheminformatics](https://practicalcheminformatics.blogspot.com/2018/05/free-wilson-analysis.html) with an excellent [notebook](https://colab.research.google.com/github/PatWalters/practical_cheminformatics_tutorials/blob/main/sar_analysis/free_wilson.ipynb), and I'll be addressing a few of the edge cases that the current code can use a hand with. You can find my notebook, which is a modified version of Pat's [here]() if you'd like to follow along.

First, the edge cases. The current code struggles with molecules which have two R-groups on the same attachment point, or rings that attach to two attachment points simultaneously. I've drawn up three test cases that we use to verify that our improvements actually work:


The first thing we need to fix is the way two R-groups on the same attachment point are handled. By default RDKit groups these into one attachment, separated by a period, e.g. `C[*:3].C[*:3]`. This causes trouble when we will be molzipping these compound back together later on, because we will be using the same attachment point twice which RDKit (rightfully) doesn't appreciate. This snippet changes the default behaviour to create a second attachment point on the double-substituted attachment points:
```python
from rdkit.Chem import rdRGroupDecomposition
ps = rdRGroupDecomposition.RGroupDecompositionParameters()
ps.allowMultipleRGroupsOnUnlabelled = True

match, miss = RGroupDecompose(core_mol,df.mol.values,asSmiles=True, options=ps)
```

That fixes issues with the double-substituted compounds, but we still run into trouble if we have rings that attach to two attachment points. If we have a ring that attaches to R1 and R5, RDKit will put that moiety in both the R1 and R5. This is logical behaviour, but it doesn't play nicely with molzip, causing two related issues. Firstly, combining the ring `CCC(C[*:1])[*:5]` at R5 with another substituent at R1 doesn't really make sense - there's no molecular answer that makes sense there, so the best we can do is to skip it with the following code snippet:
```python
import re
def has_shared_number(string_tuple):
  """Checks if any number within the [*:number] format is shared among strings in a tuple.

  Args:
    string_tuple: A tuple containing strings with potential [*:number] patterns.

  Returns:
    True if any number is shared, False otherwise.
  """
  all_numbers = []
  for string in string_tuple:
    numbers = [int(match.group(1)) for match in re.finditer(r"\[\*:(\d+)\]", string)]
    all_numbers.extend(numbers)
  return len(set(all_numbers)) < len(all_numbers)

#Use this in our enumeration code
prod_list = []
for i,p in tqdm(enumerate(product(*enc.categories)),total=total_possible_products):
    core_smiles = rgroup_df.Core.values[0]
    if has_shared_number(p):
      continue

    try:
      smi = (".".join(p))
      mol = Chem.MolFromSmiles(smi+"."+core_smiles)
      prod = Chem.molzip(mol)
      prod = Chem.RemoveAllHs(prod)
      prod_smi = Chem.MolToSmiles(prod)
      if prod_smi not in already_made_smiles:
          desc = enc.transform([p])
          prod_pred_ic50 = full_model.predict(desc)[0]
          prod_list.append([prod_smi,prod_pred_ic50])
    except:
        print(p)
        break

```
This works, but it doesn't let us use the fused ring in our enumerations anymore - because it appears twice on R1 and R5, it's also filtered out, even though it _could_ correspond to a sensible molecule here. Rewriting the code to remove the duplicate in the smiles joining step avoids this issue.

```python
prod_list = []
for i,p in tqdm(enumerate(product(*enc.categories)),total=total_possible_products):
    core_smiles = rgroup_df.Core.values[0]
    if has_shared_number(p):
      if len(set(p)) != len(p):
        smi = (".".join(list(set(p))))
      else:
        continue
    else:
      smi = (".".join(p))
    mol = Chem.MolFromSmiles(smi+"."+core_smiles)
    prod = Chem.molzip(mol)
    prod = Chem.RemoveAllHs(prod)
    prod_smi = Chem.MolToSmiles(prod)
    if prod_smi not in already_made_smiles:
        desc = enc.transform([p])
        prod_pred_ic50 = full_model.predict(desc)[0]
        prod_list.append([prod_smi,prod_pred_ic50])
```
And there we have it! Looking at our final enumeration results, we can see that the enumeration now finishes without failures and we can find our fused systems back in the results:



