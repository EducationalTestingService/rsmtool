# Description of RSMTool feature file

This file contains information about what features and transformations should be used for scoring model building. No feature file is necessary for models with automatic feature selection.

The file must be in `.json` format and have the following structure:

```json
    {
        "features": [{
                         Information about Feature 1
                     },
                     {
                         Information about Feature 2
                     }]
    }
```


## Fields required for each feature

`feature`: the name of the feature. This must match the feature name in the data file exactly, including capitalization. Feature names should not contain hyphens. The following features names are not permitted: `spkitemid`, `spkitemlab`, `itemType`, `r1`, `r2`, `score`, `sc`, `sc1`, and `adj`.

`transform` - transformation that should be applied to the feature values. 

Possible values are:

    * raw - no transformation, use original feature value

    * org - same as raw

    * inv - 1/x

    * sqrt - square root

    * addOneInv - 1/(x+1)

    * addOneLn - ln(x+1)

Note that `rsmtool` will return an error if the values in the data do not allow the supplied transformation (for example, `inv` is applied to feature which contains 0 values). If you really want to use the tranformation, you need to pre-process your data files to remove the problematic cases. 

`sign` - after transformation, each feature value will be multiplied by this number. This field is usually set to `1` or `-1` depending on the expected sign of the correlation between transformed feature and human score to ensure that all features in the final models have positive correlation with the score. 
When determining the sign, you should take into account both the original correlation between the feature and the score and the applied transformation.  For example, if you use feature which is has negative correlation with human score and apply `sqrt` transformation, `sign` should be set to `-1`. However, if you use the same feature but apply `inv` transformation, `sign` should be set to `1`.

You can check the sign of correlations for both raw and processed features in the final report. 


