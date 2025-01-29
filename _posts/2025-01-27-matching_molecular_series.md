---
layout: post
title: "Matching Molecular Series with MatchMolSeries.py"
date: 2025-01-27
---
"What else can we put on that position to improve potency?" If you've worked in small molecule discovery, you'll probably have heard some variation of 
that question before. Replacing R-groups tends to be synthetically much easier than modifying the scaffold of a compound, and it can often dramatically
modulate not only the potency of a compound but also its other properties. It should come as no surprise then that executing libraries of R-group variations
has become a staple of modern drug discovery. Such libraries result in rich data sets where a position is exhaustively varied but everything else is kept
constant. Matched Molecular Series (MMS) were introduced by [Wawer and Bajorath](https://pubs.acs.org/doi/10.1021/jm200026b) in 2011 as a way to transfer knowledge from existing datasets to accelerate discovery.
In this post we'll be implementing Matched Molecular Series in python.

I worked together on this little project with my good friend (and software developer/data engineer) Alex Rosier. 
First things first, the actual code can be found [here](https://github.com/driesvr/MatchMolSeries). The logic behind MMS is well described in the paper linked above, but the broad strokes of it are fairly intuitive.
Let’s assume you’re a medical chemist working on a new series. You make a few analogues on a given position. The data comes back and you notice that the SAR is really similar to that of another set of compounds you worked once upon a time for a different target. Intriguing! You dust off your old ELN, looking for the most potent substituents you made back then and put them into synthesis. 

The above is in a nutshell what MMS does, except MMS enables you to do it systematically. By finding other compound series where the SAR on a given position appears to track similarly (typically on different targets and/or scaffolds) we can use those other series to decide on what other groups we can try on a given position of our own scaffold.

On a practical level, we can break it down into a few steps.
- Break apart all compounds into cores and fragments 
- Store cores, fragments and assay data 
- Execute the above two steps for both query and reference compounds
- Match fragments between reference and query datasets 
- Group those by core+assay and filter for a given number of fragments that need to match
- From that filtered set take any additional un-matched fragments
- Those unmatched fragments are your new R-groups to try!


Before we really dive into the actual MMS, it's a good idea to add some code to standardize all our molecules. This will help make sure we don't miss anything due to different ways of representing similar molecules. The code below does a decent job, but note that we are _not_ checking for tautomers here. Depending on the dataset, you may want to add tautomer canonicalization.

```python
    def _standardize_mol(mol: Chem.Mol, frag_remover: rdMolStandardize.FragmentRemover) -> Chem.Mol:
        """
        Standardize a molecule using RDKit standardization methods.
        
        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule to standardize.
        
        Returns
        -------
        Chem.Mol
            Standardized RDKit molecule.
        """
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)
        mol = rdMolStandardize.Normalize(mol)
        mol = rdMolStandardize.Reionize(mol)
        mol = frag_remover.remove(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        return mol
```

Now that we have a way to standardize our molecules, let's take a look at breaking them down into cores and fragments. I'm using the following two RDKit reaction SMARTS patterns:
```python
splitting_reactions = [
            AllChem.ReactionFromSmarts('[*;R:1]-!@[*:2]>>[*:1][At].[At][*:2]'),
            AllChem.ReactionFromSmarts('[!#6;!R:1]-!@[C;!X3;!R:2]>>[*:1][At].[At][*:2]')
        ]
```
These reactions should match the [Matsy](https://pubs.acs.org/doi/10.1021/jm500022q) implementation and break single acyclic bonds between either a ring atom and any other atom, 
or a heteroatom bonded to a non-sp2 C aton. I'm terminating the bonds we broke with an `At` atom. That's somewhat arbitrary, but I like `At` here because it gives you _technically_ valid molecules that can be read into most cheminformatics software, 
it's easy to recognize (at least, until someone tries to put one in a drug) and you could even imagine it to be short for "attachment point". You could also terminate with dummy atoms. We can of course consider other fragmentation options here, like the classic 
[Hussain-Rea](https://pubs.acs.org/doi/10.1021/ci900450m) implementation that splits at any acyclic bond. Any fragmentation method will work here. As long as it splits the molecule into two parts,
downstream steps will be the same. While we're at it, let's also define a reaction to combine the two fragments back together:

```python
combine_reaction = AllChem.ReactionFromSmarts('[*:1][At].[At][*:2]>>[*:1]-[*:2]')
```

The above provides us with the necessary machinery to start splitting our molecules into cores and fragments. Starting from a pandas dataframe with smiles, potency and assay values, 
we can loop over it and store all the information we will need in subsequent steps. We're also introducing a few parameters to control the behavior of the method:
- `query_or_ref`: Specify whether this is a reference or query set
- `max_mol_atomcount`: controls the maximum number of atoms allowed in a molecule before we skip it
- `max_fragsize_fraction`: controls the maximum size of fragment relative to parent molecule

Note that the function will return a polars dataframe, rather than the pandas dataframe that went in. We're using polars because it will enable us to accelerate the queries we'll be performing in later steps.
We're also differentiating between reference and query datasets in order to make the column names clearer.

```python
    def fragment_molecules(self, input_df: pd.DataFrame, query_or_ref: str = 'ref',
                         smiles_col: str = 'smiles', potency_col: str = 'potency',
                         assay_col: str = 'assay', max_mol_atomcount: float = 100,
                         standardize: bool = True, 
                         max_fragsize_fraction: float = 0.5) -> pl.DataFrame:
        """
        Fragment molecules from an input DataFrame using SMARTS-based chemical transformations.
        
        Parameters
        ----------
        input_df : pandas.DataFrame
            Input DataFrame containing molecule information
        query_or_ref : {'ref', 'query'}, default='ref'
            Specify whether this is a reference or query set
        smiles_col : str, default='smiles'
            Name of column containing SMILES strings
        potency_col : str, default='potency'
            Name of column containing potency values
        assay_col : str, default='assay'
            Name of column containing assay identifiers
        max_mol_atomcount : float, default=100
            Maximum number of atoms allowed in a molecule
        standardize : bool, default=True
            Whether to standardize molecules using RDKit's StandardizeMol
        max_fragsize_fraction : float, default=0.5
            Maximum allowed size of fragment relative to parent molecule (0.0-1.0)
            
        Returns
        -------
        polars.DataFrame
            DataFrame containing fragment information. 
            DataFrame is additionally stored in self.fragments_df or self.query_fragments_df
            
        Raises
        ------
        ValueError
            If input DataFrame is empty or missing required columns
        """
        if input_df.empty:
            raise ValueError("Input DataFrame is empty")
        if not all(col in input_df.columns for col in [smiles_col, potency_col, assay_col]):
            raise ValueError(f"Missing required columns: {smiles_col}, {potency_col}, or {assay_col}")

        frag_remover = rdMolStandardize.FragmentRemover()

        # Initialize lists to store fragment information
        fragment_smiles_list = []
        molecule_indices = []
        cut_indices = []
        fragment_potencies = []
        fragment_sizes = []
        parent_smiles_list = []
        core_smiles_list = []
        assay_list = []
        
        # Process each molecule
        input_df = input_df.sort_values(by=assay_col, ascending=False)
        for mol_idx, (smiles, potency, assay) in enumerate(input_df[[smiles_col, potency_col, assay_col]].values):
            if (mol_idx % 1000 == 0) & (mol_idx>0):
                logger.info(f"Processed {mol_idx} molecules...")
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                logger.warning(f'Molecule {mol_idx} is None')
                continue
            
            if standardize:
                try:
                    mol = self._standardize_mol(mol, frag_remover)
                except Exception as e:
                    logger.error(f'Failed to standardize molecule {mol_idx}: {e}')
                    continue
            
            parent_atom_count = mol.GetNumHeavyAtoms()
            if parent_atom_count > max_mol_atomcount:
                continue
            
            # Apply transformations
            products = []
            for rxn in self.splitting_reactions:
                products.extend(rxn.RunReactants((mol,)))
            
            # Process each product
            for cut_idx, frags in enumerate(products):
                if not frags or len(frags) != 2:
                    continue
                    
                try:
                    [Chem.SanitizeMol(frag) for frag in frags]
                except:
                    continue
                
                # Get sizes and determine core/fragment
                frag1_size = frags[0].GetNumHeavyAtoms() - 1
                frag2_size = frags[1].GetNumHeavyAtoms() - 1
                
                if frag1_size >= frag2_size:
                    core, fragment = frags[0], frags[1]
                else:
                    core, fragment = frags[1], frags[0]
                
                # Get SMILES
                core_smiles = Chem.MolToSmiles(core, canonical=True)
                fragment_smiles = Chem.MolToSmiles(fragment, canonical=True)
                
                # Check fragment size
                fragment_size = (fragment.GetNumHeavyAtoms() - 1) / parent_atom_count 
                if fragment_size > max_fragsize_fraction:
                    continue
                
                # Store data
                fragment_smiles_list.append(fragment_smiles)
                molecule_indices.append(mol_idx)
                core_smiles_list.append(core_smiles)
                cut_indices.append(cut_idx)
                fragment_potencies.append(potency)
                fragment_sizes.append(round(fragment_size, 1))
                parent_smiles_list.append(smiles)
                assay_list.append(assay)
        
        # Create output DataFrame
        if query_or_ref == 'ref':
            fragments_df = pl.DataFrame({
                'id': range(len(fragment_smiles_list)),
                'fragment_smiles': fragment_smiles_list,
                'molecule_idx': molecule_indices, 
                'cut_idx': cut_indices,
                'parent_potency': fragment_potencies,
                'fragment_size': fragment_sizes,
                'parent_smiles': parent_smiles_list,
                'ref_core': core_smiles_list,
                'ref_assay': assay_list
            })
        else:
            fragments_df = pl.DataFrame({
                'id': range(len(fragment_smiles_list)),
                'fragment_smiles': fragment_smiles_list,
                'molecule_idx': molecule_indices, 
                'cut_idx': cut_indices,
                'parent_potency': fragment_potencies,
                'fragment_size': fragment_sizes,
                'parent_smiles': parent_smiles_list,
                'core_smiles': core_smiles_list,
                'assay': assay_list
            })
        
        fragments_df = fragments_df.unique(subset=['parent_smiles', 'fragment_smiles'] + 
                                        (['ref_assay'] if query_or_ref == 'ref' else ['assay']))

        # Store DataFrame
        if query_or_ref == 'query':
            self.query_fragments_df = fragments_df
        else:
            self.fragments_df = fragments_df
            
        return fragments_df

```

That's a pretty good start! After we run this on a reference dataset - I used a subset of BindingDB_Patents -  we'll have a dataframe that we can query. 
You'll see that we start by converting our polars dataframes into lazy mode (aka a lazyframe). This unlocks a bunch of optimisations that we use to our advantage, notably 
automatic query optimisation, streaming for larger-than-memory datasets and upfront schema error detection. This does entail we won't have any of the intermediate results as we would with pandas. 

Coming back to the workflow we outlined at the beginning, we will need to write code to accomplish these three steps:
- Match fragments between reference and query datasets 
- Group those by core+assay and filter for a given number of fragments that need to match
- From that filtered set take any additional un-matched fragments

We start off by filtering out anything in either the reference or query dataset that has less than `min_series_length` fragments: anything that doesn't belong to a long enough series can be safely discarded.
We then merge reference and query datasets on their fragment SMILES, identifying fragments present in both datasets. That by itself isn't sufficient: we still need to make sure they belong to a long enough series. 
To do that, we group by core+assay (for both reference and query) and count the number of fragments in each series. We then filter out any series with less than `min_series_length` fragments. 
In that same step, we also compute the cRMSD (a metric proposed by [Ehmki and Kramer](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00709) to track the similarity between the two series) between the potency vectors of the reference and query series. 
It provides an indication of how well trends align between the two series, with lower values indicating better similarity (and a cRMSD of 0 indicating that the SAR tracks perfectly). It's computed as follows:

\\\[
\text{cRMSD} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left[ (x_i - \bar{x}) - (y_i - \bar{y}) \right]^2}
\\\]

We also track fragments in common and the potencies in both assay sets, here by casting them to a string and joining them with a pipe character. 
Finally, we left join the reference and query datasets again on their fragment smiles, but select only those where the query fragment is missing: this set of un-matched fragments will contain - among other things - our new R-groups
of interest. By grouping on assay+core again and merging that with the matched series dataframe, we make sure we only retain R-groups that actually belong to the same matched series, 
bringing us to our final output: a dataframe where each row holds a matched series, along with all the relevant information.

The optimisations provided by the polars library makes this entire process pretty snappy. On my local machine I can match 500 compounds to the 600K compounds reference dataset in just under 10 seconds. 



```python
    def query_fragments(self, query_dataset: pd.DataFrame, min_series_length: int = 3, 
                       assay_col: str = 'assay', standardize: bool = True,
                       fragments_already_processed: bool = False,
                       max_fragsize_fraction: float = 0.5) -> pl.DataFrame:
        """
        Query the fragment database using a dataset of molecules.
        
        Parameters
        ----------
        query_dataset : pandas.DataFrame
            Dataset containing query molecules
        min_series_length : int, default=3
            Minimum number of fragments required to consider a series
        assay_col : str, default='assay'
            Name of column containing assay identifiers
        standardize : bool, default=True
            Whether to standardize molecules using RDKit standardization methods
        max_fragsize_fraction : float, default=0.5
            Maximum allowed size of fragment relative to parent molecule
        fragments_already_processed : bool, default=False
            Whether the input file contains fragments that have already been processed(e.g. originate from this class)
        Returns
        -------
        polars.DataFrame
            DataFrame containing matched series information
        """
        # Get query fragments
        if fragments_already_processed:
            self.query_fragments_df = pl.from_pandas(query_dataset) if not isinstance(query_dataset, pl.DataFrame) else query_dataset
        else:
            self.fragment_molecules(query_dataset, assay_col=assay_col, query_or_ref='query',
                              max_fragsize_fraction=max_fragsize_fraction, standardize=standardize)

        # Convert to lazy DataFrames
        reference_fragments_lazy = self.fragments_df.lazy()
        query_fragments_lazy = self.query_fragments_df.lazy()

        # Filter and join fragments
        reference_series = (reference_fragments_lazy
            .group_by(['ref_core', 'ref_assay'])
            .agg(pl.n_unique('fragment_smiles').alias('reference_fragment_count'))
            .filter(pl.col('reference_fragment_count') >= min_series_length)
            .join(reference_fragments_lazy, on=['ref_core', 'ref_assay'])
        )

        query_series = (query_fragments_lazy
            .group_by(['core_smiles', 'assay'])
            .agg(pl.n_unique('fragment_smiles').alias('query_fragment_count'))
            .filter(pl.col('query_fragment_count') >= min_series_length)
            .join(query_fragments_lazy, on=['core_smiles', 'assay'])
        )

        # Join and process results
        merged_series = reference_series.join(query_series, on='fragment_smiles')

        matched_series = (merged_series
            .group_by(['ref_core', 'ref_assay', 'core_smiles', 'assay'])  
            .agg([
                pl.n_unique('fragment_smiles').alias('series_length'),
                pl.first('core_smiles').alias('query_core'),
                pl.first('assay').alias('query_assay'),
                pl.col('fragment_smiles').alias('common_fragments').str.join('|'),
                pl.col('parent_potency').cast(pl.Utf8).str.join('|').alias('reference_potencies'),
                pl.col('parent_potency_right').cast(pl.Utf8).str.join('|').alias('query_potencies'),
                (pl.col('parent_potency') * pl.col('parent_potency_right')).sum().alias('potency_dot_product'),
                (pl.col('parent_potency') ** 2).sum().alias('reference_potency_norm_sq'),
                (pl.col('parent_potency_right') ** 2).sum().alias('query_potency_norm_sq'),
                ((pl.col('parent_potency') - pl.col('parent_potency').mean() - 
                (pl.col('parent_potency_right') - pl.col('parent_potency_right').mean()))**2).mean().sqrt().alias('cRMSD')
            ])
            .filter(pl.col('series_length') >= min_series_length)
        )

        # Find additional fragments in reference set not present in query
        unique_reference_fragments = (reference_series
            .join(query_series, on='fragment_smiles', how='left')
            .filter(pl.col('id_right').is_null())
            .select(['fragment_smiles', 'ref_core', 'ref_assay', 'parent_potency'])
            .group_by(['ref_core', 'ref_assay'])
            .agg([
                pl.col('fragment_smiles').str.join('|').alias('new_fragments'),
                pl.col('parent_potency').cast(pl.Utf8).str.join('|').alias('new_fragments_ref_potency')
            ])
            .join(matched_series, on=['ref_core', 'ref_assay'])
            .select([
                'new_fragments', 
                'ref_core', 
                'query_core', 
                'ref_assay', 
                'query_assay', 
                'cRMSD',
                'series_length', 
                'common_fragments',
                'reference_potencies',
                'query_potencies',
                'new_fragments_ref_potency'
            ])
        )
        
        result = unique_reference_fragments.collect().to_pandas()
        result['ref_core_attachment_point'] = result['ref_core'].apply(self._get_attachment_point)
        result['query_core_attachment_point'] = result['query_core'].apply(self._get_attachment_point)
        return result
```

Right before we return the dataframe, we add some additional columns containing the attachment points of the R-group series on the core molecules to make the results more interpretable:
self._get_attachment_point. As it's clearly marked with the `[At]` atomtype, we can just use `match = mol.GetSubstructMatch(Chem.MolFromSmarts('[*:1][At]'))` to find the attachment point followed by 
`mol.GetAtomWithIdx(match[0]).GetSmarts()` to get the SMARTS representation of the atom. 

If all went well, we should now have a working implementation of matched molecular series. There's some additional QOL functionality in the  [package itself](https://github.com/driesvr/MatchMolSeries), but the key
functionality was covered in this post. All that's left now is to test everything works and see if we can get some interesting suggestions for the next compounds to try!

```python
    def test_concatenation_order(self):
        """
        Verifies that the system correctly concatenates
        molecules and their respective potency values
        
        """
        ref_data = pd.DataFrame({
            'smiles': ['c1ccccc1F', 'c1ccccc1Cl', 'c1ccccc1Br','c1ccccc1N', 'c1ccccc1OC(F)(F)F'],
            'potency': [1.0, 2.0, 3.0, 4.0, 5.0],
            'assay_col': ['assay1']*5
        })
        
        query_data = pd.DataFrame({
            'smiles': ['c1cnccc1F', 'c1cnccc1Cl', 'c1cnccc1Br'],
            'potency': [1.0, 2.0, 3.0],
            'assay_col': ['assay1']*3
        })
        self.mms.fragment_molecules(ref_data, assay_col='assay_col', query_or_ref='ref')
        result = self.mms.query_fragments(query_data, min_series_length=3, assay_col='assay_col')
        new_frags = result.new_fragments[0].split('|')
        ref_potency = result.new_fragments_ref_potency[0].split('|')
        print(new_frags, ref_potency)
        self.assertEqual(new_frags.index('N[At]'), ref_potency.index('4.0')) 
        self.assertEqual(new_frags.index('FC(F)(F)O[At]'), ref_potency.index('5.0')) 

```
This test will check that the two additional R-groups in the reference dataset (in this case, an aniline and a trifluoromethoxy) can be retrieved based on the matching series of 
F, Br and Cl substituents. It also checks that the potency values are associated with the correct R-groups. This finishes succesfully on my setup, so it looks like that all works! Let's see what we can retrieve when we check for such series in the bindingDB IC50 patent dataset. First, we put together a series of molecules that gains about a threefold in potency as we move to heavier halogens.

```python
query_df = pd.DataFrame({
    'smiles': ['c1ccccc1F', 'c1ccccc1Cl', 'c1ccccc1Br'],
    'potency': [6.5, 7.0, 7.5],  
    'assay_col': ['assay1', 'assay1', 'assay1']
})
```

This query is a short series and as such there's going to bea lot of matches. We'll take a look at one of them in more detail. This series comes from patent US8765744, targeting 11-beta-hydroxysteroid dehydrogenase 1, 
perhaps better known as cortisone reductase. The cRMSD of this series is 0.067, indicating that the trends in potency between the reference and query series are very similar: the potencies are 6.48 for the 
F analog, 7.14 for the Cl and 7.59 for the Br. That gives us some confidence that the series is indeed a good candidate for SAR transfer and we can look at the additional R-groups in the reference dataset:

| R-group SMARTS | Value |
|----------------|-------|
| FC(F)O[At] | 7.09 |
| Cn1ccc([At])cc1=O | 6.74 |
| FC(F)(F)[At] | 7.09 |
| [At]C1CC1 | 7.64 |
| N#C[At] | 6.18 |
| C[At] | 7.19 |
| CO[At] | 7.46 |
| CC(C)(C)[At] | 6.84 |
| FC(F)[At] | 7.06 |
| FC(F)(F)O[At] | 7.24 |

Browsing through the structures of the R-groups, we can see that they mostly are predominantly small apolar groups that wouldn't look out of place in most medchem programs. While none of them are substantially more potent than the bromine, the cyclopropyl and methoxy are roughly equipotent to the bromine and may be an interesting candidate for our hypothetical series. 

That's it for today! We went through the basic concepts of matched molecular series, implemented a prototype in Python, and tested it on a small dataset. I'll close off this post with some recommended reading on MMS: [Original MMS paper](https://pubs.acs.org/doi/10.1021/jm200026b), [Ehmki and Kramer on metrics for SAR transfer](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00709),
[Matsy paper](https://pubs.acs.org/doi/10.1021/jm500022q), and [MMS for ADME](https://pubs.acs.org/doi/full/10.1021/acs.jcim.0c00269). I hope you found this post interesting and as always,
please let me know if you spot any mistakes (or better yet, submit a PR to fix them)!
