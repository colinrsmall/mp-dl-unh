# mp-dl-unh
This project is a part of the [MMS mission](https://lasp.colorado.edu/mms/sdc/public/) at the University of Colorado-Boulder's [Laboratory for Atmospheric and Space Physics](http://lasp.colorado.edu/home/) and the University of New Hampshire's [Space Science Center](https://eos.unh.edu/space-science-center). 

## Project Intro/Objective
The mp-dl-unh software is designed to provide automated magnetopause crossing selection suggestions for the SITL selection team. The software uses a [TensorFlow Keras](https://www.tensorflow.org/guide/keras) [LSTM (long short term memory) neurel network](https://en.wikipedia.org/wiki/Long_short-term_memory) trained on previous magnetopause selections in order to generate suggested crossing selection windows.

### Partners
* [Matthew R. Argall](https://mypages.unh.edu/argallmr/bio), [University of New Hampshire Space Science Center](https://eos.unh.edu/space-science-center)
* [Marek Petrik](https://ceps.unh.edu/person/marek-petrik), [University of New Hampshire Department of Computer Science](https://ceps.unh.edu/computer-science)

## Detailed Project & Software Description

### Dependencies
* Numpy
* Pandas
* Scipy
* Keras
* TensorFlow
* Spacepy
