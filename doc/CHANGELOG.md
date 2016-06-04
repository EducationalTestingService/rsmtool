# Release Notes

## v5.0.0 (June 3, 2016)

### New features

* Evaluations on the test set now include R2 and RMSE.

* The tool now reports model fit parameters (R2 and adjusted R2) for the training set.

* It is now possible to exclude candidates with less than X responses from model training/evaluation.

* RSMCompare can now handle experiments which used SKLL models.

* RSMCompare now includes a notebook for consistency between human raters.

### Main bug fixes

* Correct handling of repeated feature names in the feature .json file.

* Correct printing of feature coefficients for SKLL models.

* Correct handling of quoted boolean values in config .json file.

* Fixed rounding and highlighting in feature correlation table.

## v4.6.0 (April 4, 2016)

First github release. 
