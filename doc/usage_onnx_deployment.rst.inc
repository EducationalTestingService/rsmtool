.. _usage_onnx_deployment:

Deploy RSMTool models
^^^^^^^^^^^^^^^^^^^^^

RSMTool depends on many large python libraries which can make it tricky to efficiently deploy the trained models. This example `deploy_as_onnx.py <https://github.com/EducationalTestingService/rsmtool/blob/main/examples/deploy_as_onnx.py>`_ demonstrates how to export a simple RSMTool model to ONNX. The resulting model depends only on ``onnxruntime`` and ``numpy``.

Pre- and post-processing
""""""""""""""""""""""""

The example script supports many pre-processing steps of RSMTool, such as, clipping outliers and z-normalization, and also post-processing steps, such as, scaling and clipping predictions. These steps are done in numpy, before and after calling the ONNX model. While not all features of RSMTool are supported, many of them could be supported by adjusting the numpy pre- and post-processing code.

Model export
""""""""""""

In this example, we use `skl2onnx https://pypi.org/project/skl2onnx/`_ to export the underlying scikit-learn model to ONNX. Should this process fail, it is possible to export the scikit-learn model with ``joblib`` (``scikit-learn`` will then be a runtime dependecy).
