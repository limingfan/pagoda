# Pagoda README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Pagoda is one GPU runtime system which virtualizes GPU hardware resources to increase the utilization of GPU in narrow task workloads. 
* 1.0

### How do I get set up? ###

* Install CUDA library >= 7.5 and python >=2.6
* Configuration

  1.  To change the value of SM_NUM in common/para.h based on your GPU. 
   
  2. To change the path of CUDA and the value of sm (e.g. sm_35 or sm_52 ...) in common/make.config

* Dependencies
* Database configuration
* How to run tests

  1. To execute ppopp17 script after setting of the configuration
  
  2. The run script file in each benchmark folder

* Deployment instructions

### Contribution guidelines ###

* Writing tests

  1. People can follow Pagoda APIs used in pagoda folders in each benchmark to work out your own pagoda programs. 

* Code review

  1. baseline: this folder contains proglems composed by standard CUDA APIs

  2. pagoda : this folder includes programs that consist of Pagoda APIs

* Other guidelines

### Who do I talk to? ###

* Contact: yeh14@purdue.edu