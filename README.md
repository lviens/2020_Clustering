# Viens L. and Iwata T. (2020), Improving the retrieval of offshore-onshore correlation functions with machine learning, Accepted in JGR - Solid Earth

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

- Update - 21/07/2020: The paper has been accepted in JGR - Solid Earth ([The accepted paper is available on EarthArXiv](https://eartharxiv.org/8ba5p/)).
- Update - 03/06/2020: 2nd release of the code.
- 06/03/2020: 1st release of the code

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

## Description:
We developped a method based on unsupervised learning to improve the retrieval of deconvolution functions between offshore and onshore stations. We applied our method to retrieve deconvolution functions between offshore seismic stations deployed on top of the Nankai, Japan, and surrounding onshore stations. 

* This repo contains three folders:
  * **Codes** folder:
    - **Reproduce_Fig_1_2.py** to reproduce Figures 1 and 2 of the paper.
    - **Reproduce_Fig_4_5.py** to reproduce Figures 4 and 5 of the paper (data download required, see below).
    - **Function_clustering_DFs.py**, the main function to perform the clustering (used by the two codes above).

  * **Data** folder:
    - Empty folder. The data required to reproduce Figures 4 and 5 can be downloaded [here](https://drive.google.com/file/d/1wbM-cN4gQ-MRhLOQaiXcHiXZ5Z5OOEsI/view?usp=sharing) and should be placed in the **Data** folder.

  * **Figures** folder:
    - Contains the 4 figures generated by the two codes.

## **Additional info**:
  * The deconvolution functions in the paper are computed using the deconvolution_stab function available [here](https://github.com/lviens/2017_GJI/blob/master/Codes/Functions_GJI_2017.py).

  * The raw Hi-net and DONET 1 data can be downloaded [here](http://www.hinet.bosai.go.jp).
