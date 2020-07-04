# MTF_Estimator
Robust ESF, PSF, FWHM &amp; MTF estimation from low quality images.

## Introduction

The Modulation Transfer Function (MTF) of an optical system represents the spatial frequency performance of such system. Its behaviour is directly related to the concepts of <i>sharpness</i> and <i>spatial resolution</i>. For this reason it is an important quality indicator, hence the need of an estimation method.
The MTF can be inferred from the optical system design but, unfortunately, in the real world only its modulus can be empirically determined or measured from the actual system.

One way to perform the estimation is using actual images produced by the system. For example, the <a href='https://www.iso.org/standard/71696.html'>ISO 12233</a> provides an estimation standard focused on still picture cameras using <a href='https://www.imatest.com/solutions/iso-12233/'>test charts</a> or standarized targets. There are multiple ways of estimating the MTF using images, all of them based on the same idea: Extracting a range of spatial frequency information beyond the discrete values that the image directly provides.



