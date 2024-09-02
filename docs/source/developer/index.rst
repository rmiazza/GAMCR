Developer
=========

You are interested by scClone2DR and you want to use our code for your own research projects? This is the place to be ! 

In that case, you may require to modify the code to fit your specific expectations. In this section, we provide additional information on our implementation that should allow you to easily modify the code for your needs (and in particular to add new functionalities).


Features at the subclone level
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

scClone2DR can be used by specifying yourself the features at the subclone level. In the case where features are computed as the single cell level, scClone2DR will aggregate the features among all cells belonging to a given clone by learning an attention mechanism, as originally proposed in :doc:`Engelmann2024 <../references>`.

How can I learn feature vectors for all cells from single cell multi-omics data?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use your favourite method to achieve this goal. To guide us identify the method that would be the more suitable for your use case, we provided in the appendix of our paper a review of the literature for Variational Auto-Encoder (VAE) based method for the integration of single cell multi-omics data. In this review of the literature, we shed light on the method that provide interpretable latent representation of the high dimensional single cell measurements. 


How scClone2DR is trained?
^^^^^^^^^^^^^^^^^^^^^^^^^^

When the feature vectors are directly provided at the level of subclones, scClone2DR does not contain any latent variables and parameter can be estimated from maximum likelihood. The log-likelihood being not convex, the model parameters are learned using a stachastic gradient algorithm.

When the feature vectors are defined at the single cell level, scClone2DR contain some latent variables, being the parameters of the attention meachanisme allowing to aggregate feature vector of the subclonal level. Learning the parameters of scClone2DR by maximizing the posterior distribution is intractable. To bypass this issue, we rely on variational inference and we approximate the posterior distribution by a multivariate normal distribution. We then minimize the so-called ELBO (Evidence Lower Bound) which consists in optimizing jointly over the model parameters and the parameters of the variational distribution. For this, we rely on a stochatic variational inference algorithm implemented in the python package Pyro.



Clones and cells || Types, labels and categories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: /_static/clone_label_type.png
  :width: 800
  :alt: Hierarchy of attributes given to single cells and clones.


- Each cell is assigned to a clone.
- One can assign a type to each clone. Typically, the celltype the most present in the clone is considered. Note that this clone-type assignment is an intermediate step to define clonel labels but is not used in scClone2DR.
- Then, one can define the labels to clone. In the extreme cases: 1) one can define only two labels (healthy and tumor) and thus labels will be equal to categories, or 2) one can give a different label to all subclones. 
- The model considers two categories for subclones: healthy and tumor group of cells.


The clone labels will be used by scClone2DR to make some parameters label specific. Typically, for AML, one can define among the different groups of tumor cells some of them that are "putative" or directly "tumor". In this case, scClone2DR will compute the survival probabilitie :math:`\pi_{idk}` for any subclone k by using parameters that are label specific:

.. math::
   \pi_{idk}:=\sigma(\textbf x_{ik}(\gamma_{\ell_k})^{\top}\mathbf \beta_d + b_{d}^ {\ell_k})
 

where :math:`\ell_k` is the label for the subclone number k, :math:`b_{d}^{\ell_k}` is the offset for drug d and subclone with label :math:`\ell_k` and :math:`\gamma_{\ell_k}` is the parameter of the attention mechanism used to obtain :math:`\textbf x_{ik}`, i.e. to aggregate the features at the single cell level to the subclone level.


