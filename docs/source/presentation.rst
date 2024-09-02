Presentation of the project
---------------------------



What is Oncology?
~~~~~~~~~~~~~~~~~

Oncology is the branch of medicine that focuses on the study, diagnosis, treatment, and prevention of cancer. Oncologists, the medical professionals in this field, work to understand the biology of cancer, develop treatment plans, and manage patient care. Oncology encompasses various sub-specialties, including medical, surgical, and radiation oncology, each targeting different aspects of cancer treatment. The goal is to improve patient outcomes through personalized therapies, research, and advancements in technology.


What is cancer pharmacology?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The specific field of research that focuses on the drug response of tumor cells is known as **pharmacogenomics in oncology** or **cancer pharmacology**. This area studies how genetic variations in tumor cells influence their response to different drugs, aiming to identify the most effective treatments with the least side effects. Researchers in this field analyze the interactions between cancer drugs and cellular pathways, resistance mechanisms, and biomarkers to develop targeted therapies that can improve patient outcomes by tailoring treatment to the genetic profile of the tumor.


What are the challenges in cancer pharmacology?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Challenges in cancer pharmacology include the complexity of tumor heterogeneity, where different cells within the same tumor can have distinct genetic and molecular profiles. This diversity makes it difficult to predict how a tumor will respond to treatment, as some cells may be resistant to certain drugs while others are sensitive. To address this, researchers are increasingly using single-cell measurements to analyze the drug response at an individual cell level. This approach helps identify specific subpopulations of cells that drive resistance, allowing for the development of more precise and effective therapies that can target these resistant cells within the heterogeneous tumor.

Other challenges in cancer pharmacology include the development of drug resistance, where tumor cells adapt over time and become less responsive to treatments. This can lead to cancer recurrence and progression. Additionally, the toxicity of cancer drugs poses a significant challenge, as treatments that are effective against cancer cells can also harm healthy cells, leading to severe side effects. The complexity of the tumor microenvironment, which includes interactions between cancer cells, immune cells, and surrounding tissues, also complicates drug efficacy. Finally, translating findings from preclinical studies to successful clinical outcomes remains difficult, as results in laboratory models do not always predict patient responses.



What are our main contributions?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coupling state of the art methods for integration of single cell multi-omics data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Integration of single cell measurements (potentially coming from mutiple modalities) is challenging. A huge effort has been dedicated to improve this task in the recent years, in particular the use of VAE-based methods. We refer to the appendix of our publication for a review of such approaches. In our paper, we first make use of SCATrEX from  :doc:`Ferreira2022 <references>`. to learn group of cells from both scDNA and scRNA-seq data. 

We also rely on a VAE-based method to embed the single cell measurements in a common latent space where each entry is interepretable in terms of pathways. These latent representations of single cells are then aggregated in a data driven way during trying using an attention mechanism. Note that the aggregation of the cells belonging to groups with different labels are performed using a label specific parameter for the attention mechanism. 


Accounting for technical noise in the experimental design
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We take for the overdispersion effect inherent to the pharmascological measurements. Moreover, we find out that other noise corrupt our data, in particular the position of the well on the plate. We account for this plate effect (together with the cell density in wells) using generalized additive models.


Relying on the generative nature of scClone2DR to inform experimental design
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our model is generative and can thus allow to generate data close to real ones. This simulated dataset can then be used to identify the set of parameters for the experimental design to ensure a reliable estimation of the model parameters. Typically, the number of replicates for treated wells is a key parameter to investigate.


A well documented and maintainable code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With this :doc:`documentation <documentation/index>` and our :doc:`tutorials <tutorials/index>`, one can easily reproduce the results of our paper. Moreover, we put a lot of effort to make our code extensible and flexible. We give a detailed description of the structure of the code in order to allow reseachers to use our code for their own projects. We give some tips to extend our package :doc:`here <developer/index>`.


