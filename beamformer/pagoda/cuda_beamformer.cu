#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "beam.h"
#include "runtime.cuh"

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

int main(){

  BeamFirData *coarse_fir_data;
  BeamFirData *fine_fir_data;
  BeamFirData *mf_fir_data;
  float **coarse_dev_weight, **coarse_dev_buffer;
  float **fine_dev_weight, **fine_dev_buffer;

  float **inputs, **inputs_dev;
  float **predec, **predec_dev;
  float **postdec, **postdec_dev;

  int *post_num_dec1, *post_num_dec1_dev;
  int *post_num_dec2, *post_num_dec2_dev;
  int *coarse_ratio, *coarse_ratio_dev;
  int *fine_ratio, *fine_ratio_dev;
  int *coarse_taps, *coarse_taps_dev;
  int *fine_taps, *fine_taps_dev;

  float **beam_weights;
  float *beam_input;
  float *beam_output;
  float *beam_fir_output;
  float *beam_fir_mag;
  float **detector_out;
 
  int i, j;

  printf("Pagoda Beamformer:#thread:%d, #task:%d\n", TDD_NUM, NUM_CHANNELS);
  setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);
  //memory allocation
  coarse_fir_data = (BeamFirData*)malloc(NUM_CHANNELS * sizeof(BeamFirData));
  fine_fir_data = (BeamFirData*)malloc(NUM_CHANNELS * sizeof(BeamFirData));
  mf_fir_data = (BeamFirData*)malloc(NUM_BEAMS * sizeof(BeamFirData));
  coarse_dev_weight = (float**)malloc(NUM_CHANNELS * sizeof(float *));
  coarse_dev_buffer = (float**)malloc(NUM_CHANNELS * sizeof(float *));
  fine_dev_weight = (float**)malloc(NUM_CHANNELS * sizeof(float *));
  fine_dev_buffer = (float**)malloc(NUM_CHANNELS * sizeof(float *));
  beam_weights = (float**)malloc(NUM_BEAMS * sizeof(float *));
  detector_out = (float**)malloc(NUM_BEAMS * sizeof(float *));

  inputs = (float**)malloc(NUM_CHANNELS * sizeof(float *));
  predec = (float**)malloc(NUM_CHANNELS * sizeof(float *));
  postdec = (float**)malloc(NUM_CHANNELS * sizeof(float *));
  inputs_dev = (float**)malloc(NUM_CHANNELS * sizeof(float *));
  predec_dev = (float**)malloc(NUM_CHANNELS * sizeof(float *));
  postdec_dev = (float**)malloc(NUM_CHANNELS * sizeof(float *));

  for(i = 0; i < NUM_CHANNELS; i++){
    checkCudaErrors(cudaHostAlloc(&inputs[i], 2*NUM_SAMPLES*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&postdec[i], 2*NUM_POST_DEC_2*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&predec[i], 2*NUM_POST_DEC_1*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc(&inputs_dev[i], 2*NUM_SAMPLES*sizeof(float)));
    checkCudaErrors(cudaMalloc(&predec_dev[i], 2*NUM_POST_DEC_1*sizeof(float)));
    checkCudaErrors(cudaMalloc(&postdec_dev[i], 2*NUM_POST_DEC_2*sizeof(float)));

    checkCudaErrors(cudaMalloc(&coarse_dev_weight[i], 2*NUM_COARSE_TAPS*sizeof(float)));
    checkCudaErrors(cudaMalloc(&coarse_dev_buffer[i], 2*NUM_COARSE_TAPS*sizeof(float)));
    checkCudaErrors(cudaMalloc(&fine_dev_weight[i], 2*NUM_FINE_TAPS*sizeof(float)));
    checkCudaErrors(cudaMalloc(&fine_dev_buffer[i], 2*NUM_FINE_TAPS*sizeof(float)));

  }

  for(i = 0; i < NUM_BEAMS; i++){
    beam_weights[i] = (float*)malloc(2*NUM_CHANNELS*sizeof(float));
    detector_out[i] = (float*)malloc(2*NUM_POST_DEC_2*sizeof(float));
  }

  beam_input = (float*)malloc(2*NUM_CHANNELS*NUM_POST_DEC_2*sizeof(float));
  beam_output = (float*)malloc(2*NUM_POST_DEC_2*sizeof(float));
  beam_fir_output = (float*)malloc(2*NUM_POST_DEC_2*sizeof(float));
  beam_fir_mag = (float*)malloc(NUM_POST_DEC_2*sizeof(float));

  checkCudaErrors(cudaHostAlloc(&post_num_dec1, sizeof(int), cudaHostAllocDefault));
  checkCudaErrors(cudaHostAlloc(&post_num_dec2, sizeof(int), cudaHostAllocDefault));
  checkCudaErrors(cudaHostAlloc(&coarse_ratio, sizeof(int), cudaHostAllocDefault));
  checkCudaErrors(cudaHostAlloc(&fine_ratio, sizeof(int), cudaHostAllocDefault));
  checkCudaErrors(cudaHostAlloc(&coarse_taps, sizeof(int), cudaHostAllocDefault));
  checkCudaErrors(cudaHostAlloc(&fine_taps, sizeof(int), cudaHostAllocDefault));
  checkCudaErrors(cudaMalloc(&post_num_dec1_dev, sizeof(int)));
  checkCudaErrors(cudaMalloc(&post_num_dec2_dev, sizeof(int)));
  checkCudaErrors(cudaMalloc(&coarse_ratio_dev, sizeof(int)));
  checkCudaErrors(cudaMalloc(&fine_ratio_dev, sizeof(int)));
  checkCudaErrors(cudaMalloc(&coarse_taps_dev, sizeof(int)));
  checkCudaErrors(cudaMalloc(&fine_taps_dev, sizeof(int)));

  *post_num_dec1 = NUM_POST_DEC_1;
  *post_num_dec2 = NUM_POST_DEC_2;
  *coarse_ratio = COARSE_DECIMATION_RATIO;
  *fine_ratio = FINE_DECIMATION_RATIO;
  *coarse_taps = NUM_COARSE_TAPS;
  *fine_taps = NUM_FINE_TAPS;


  // Init inputs
  for (i = 0; i < NUM_CHANNELS; i++)
  {
    BeamFirSetup(&coarse_fir_data[i], NUM_COARSE_TAPS);
    BeamFirSetup(&fine_fir_data[i], NUM_FINE_TAPS);
  }

  for (i = 0; i < NUM_BEAMS; i++)
  {
    BeamFirSetup(&mf_fir_data[i], MF_SIZE);
    BeamFormWeights(i, beam_weights[i]);
  }

  for (i = 0; i < NUM_CHANNELS; i++) {
    InputGenerate(i, inputs[i],
                  NUM_SAMPLES);
  }

  runtime_init();

  double start_timer, end_timer;

  //input transfer
  for (i = 0; i < NUM_CHANNELS; i++) {
    checkCudaErrors(cudaMemcpyAsync(inputs_dev[i], inputs[i], 2*NUM_SAMPLES*sizeof(float), cudaMemcpyHostToDevice, runtime_stream));
    checkCudaErrors(cudaMemcpyAsync(coarse_dev_weight[i], coarse_fir_data[i].weight, 2*NUM_COARSE_TAPS*sizeof(float), 
		cudaMemcpyHostToDevice, runtime_stream));
    checkCudaErrors(cudaMemcpyAsync(coarse_dev_buffer[i], coarse_fir_data[i].buffer, 2*NUM_COARSE_TAPS*sizeof(float), 
		cudaMemcpyHostToDevice, runtime_stream));
    checkCudaErrors(cudaMemcpyAsync(fine_dev_weight[i], fine_fir_data[i].weight, 2*NUM_FINE_TAPS*sizeof(float), 
		cudaMemcpyHostToDevice, runtime_stream));
    checkCudaErrors(cudaMemcpyAsync(fine_dev_buffer[i], fine_fir_data[i].buffer, 2*NUM_FINE_TAPS*sizeof(float), 
		cudaMemcpyHostToDevice, runtime_stream));
  }

  checkCudaErrors(cudaMemcpyAsync(post_num_dec1_dev, post_num_dec1, sizeof(int), cudaMemcpyHostToDevice, runtime_stream));
  checkCudaErrors(cudaMemcpyAsync(post_num_dec2_dev, post_num_dec2, sizeof(int), cudaMemcpyHostToDevice, runtime_stream));
  checkCudaErrors(cudaMemcpyAsync(coarse_ratio_dev, coarse_ratio, sizeof(int), cudaMemcpyHostToDevice, runtime_stream));
  checkCudaErrors(cudaMemcpyAsync(fine_ratio_dev, fine_ratio, sizeof(int), cudaMemcpyHostToDevice, runtime_stream));
  checkCudaErrors(cudaMemcpyAsync(coarse_taps_dev, coarse_taps, sizeof(int), cudaMemcpyHostToDevice, runtime_stream));
  checkCudaErrors(cudaMemcpyAsync(fine_taps_dev, fine_taps, sizeof(int), cudaMemcpyHostToDevice, runtime_stream));
  checkCudaErrors(cudaStreamSynchronize(runtime_stream));
  

  start_timer = my_timer();
#if 1
  //Beam filter
  for (i = 0; i < NUM_CHANNELS; i++) {
    taskLaunch(12, INT, TDD_NUM, INT, 1, INT, 0, INT, 0, INT, 0, INT, coarse_taps_dev, INT, post_num_dec1_dev, INT, coarse_ratio_dev,
		FLOAT, coarse_dev_weight[i], FLOAT, coarse_dev_buffer[i], FLOAT, inputs_dev[i], FLOAT, predec_dev[i]);
  }

  //checkCudaErrors(cudaDeviceSynchronize());
  waitAll(NUM_CHANNELS);

  for (i = 0; i < NUM_CHANNELS; i++) {
    taskLaunch(12, INT, TDD_NUM, INT, 1, INT, 0, INT, 0, INT, 0, INT, fine_taps_dev, INT, post_num_dec2_dev, INT, fine_ratio_dev,
                FLOAT, fine_dev_weight[i], FLOAT, fine_dev_buffer[i], FLOAT, predec_dev[i], FLOAT, postdec_dev[i]);

  }
  //checkCudaErrors(cudaDeviceSynchronize());
  waitAll(NUM_CHANNELS);
#endif
  end_timer = my_timer();
  printf("GPU elapsed Time: %lf sec.\n", end_timer - start_timer);

  start_timer = my_timer();
  for (i = 0; i < NUM_CHANNELS; i++) {
    checkCudaErrors(cudaMemcpyAsync(postdec[i], postdec_dev[i], 2*NUM_POST_DEC_2*sizeof(float), cudaMemcpyDeviceToHost, runtime_stream));
  }
  checkCudaErrors(cudaStreamSynchronize(runtime_stream));
  end_timer = my_timer();

  runtime_destroy();
  runtime_free();

#if 1
/* Assemble beam-forming input: */
  for (i = 0; i < NUM_CHANNELS; i++)
    for (j = 0; j < NUM_POST_DEC_2; j++)
    {
      beam_input[j*NUM_CHANNELS*2+2*i] = postdec[i][2*j];
      beam_input[j*NUM_CHANNELS*2+2*i+1] = postdec[i][2*j+1];
    }
  for (i = 0; i < NUM_BEAMS; i++)
  {
    /* Have now rearranged NUM_CHANNELS*NUM_POST_DEC_2 items.
     * BeamForm takes NUM_CHANNELS inputs, 2 outputs. */
    for (j = 0; j < NUM_POST_DEC_2; j++)
      BeamForm(i, beam_weights[i],
               beam_input + j*NUM_CHANNELS*2,
               beam_output + j*2);

      BeamFirFilter(&mf_fir_data[i],
                    NUM_POST_DEC_2, 1,
                    beam_output, beam_fir_output);

      Magnitude(beam_fir_output, beam_fir_mag, NUM_POST_DEC_2);
      Detector(i, beam_fir_mag, detector_out[i]);
  }
#endif
#if 0
  FILE *fp;
  fp = fopen("output2.txt", "w+");
    for (j = 0; j < NUM_POST_DEC_2; j++)
      for (i = 0; i < NUM_BEAMS; i++)
        //result = detector_out[i][j];
        //printf("%f\n", detector_out[i][j]);
        fprintf(fp, "%f\n", detector_out[i][j]);
  fclose(fp);

#endif

  //memory free

  for(i = 0; i < NUM_CHANNELS; i++){
    checkCudaErrors(cudaFreeHost(inputs[i])); 
    checkCudaErrors(cudaFreeHost(predec[i])); 
    checkCudaErrors(cudaFreeHost(postdec[i])); 
    checkCudaErrors(cudaFree(inputs_dev[i]));
    checkCudaErrors(cudaFree(predec_dev[i]));
    checkCudaErrors(cudaFree(postdec_dev[i]));

    //checkCudaErrors(cudaFreeHost(coarse_fir_data[i].weight));
    //checkCudaErrors(cudaFreeHost(coarse_fir_data[i].buffer));
    //checkCudaErrors(cudaFreeHost(fine_fir_data[i].weight));
    //checkCudaErrors(cudaFreeHost(fine_fir_data[i].buffer));

    checkCudaErrors(cudaFree(coarse_dev_weight[i]));
    checkCudaErrors(cudaFree(coarse_dev_buffer[i]));
    checkCudaErrors(cudaFree(fine_dev_weight[i]));
    checkCudaErrors(cudaFree(fine_dev_buffer[i]));

  }
  for(i = 0; i < NUM_BEAMS; i++){
    free(beam_weights[i]);
    free(detector_out[i]);
    free(mf_fir_data[i].weight);
    free(mf_fir_data[i].buffer);

  }

  checkCudaErrors(cudaFreeHost(post_num_dec1));
  checkCudaErrors(cudaFreeHost(post_num_dec2));
  checkCudaErrors(cudaFreeHost(coarse_ratio));
  checkCudaErrors(cudaFreeHost(fine_ratio));
  checkCudaErrors(cudaFreeHost(coarse_taps));
  checkCudaErrors(cudaFreeHost(fine_taps));

  checkCudaErrors(cudaFree(post_num_dec1_dev));
  checkCudaErrors(cudaFree(post_num_dec2_dev));
  checkCudaErrors(cudaFree(coarse_ratio_dev));
  checkCudaErrors(cudaFree(fine_ratio_dev));
  checkCudaErrors(cudaFree(coarse_taps_dev));
  checkCudaErrors(cudaFree(fine_taps_dev));


  free(coarse_fir_data);
  free(fine_fir_data);
  free(mf_fir_data);
  free(coarse_dev_weight);
  free(coarse_dev_buffer);
  free(fine_dev_weight);
  free(fine_dev_buffer);
  free(inputs);
  free(predec);
  free(postdec);
  free(inputs_dev);
  free(predec_dev);
  free(postdec_dev);
  free(beam_weights);
  free(beam_input);
  free(beam_output);
  free(beam_fir_output);
  free(beam_fir_mag);
  free(detector_out);

return 0;
}
void InputGenerate(int channel, float *inputs, int n)
{
  int i;
  for (i = 0; i < n; i++)
  {
    if (channel == TARGET_BEAM && i == TARGET_SAMPLE)
    {
#ifdef RANDOM_INPUTS
      inputs[2*i] = sqrt(i*channel);
      inputs[2*i+1] = sqrt(i*channel)+1;
#else
      inputs[2*i] = sqrt(CFAR_THRESHOLD);
      inputs[2*i+1] = 0;
#endif
    } else {
#ifdef RANDOM_INPUTS
      float root = sqrt(i*channel);
      inputs[2*i] = -root;
      inputs[2*i+1] = -(root+1);
#else
      inputs[2*i] = 0;
      inputs[2*i+1] = 0;
#endif
    }
  }
}

void BeamFirSetup(BeamFirData *data, int n)
{
  int i, j;

  data->len = n;
  data->count = 0;
  data->pos = 0;
  if(n == MF_SIZE){
    data->weight = (float*) malloc(sizeof(float)*2*n);
    data->buffer = (float*) malloc(sizeof(float)*2*n);
  }else{
    checkCudaErrors(cudaHostAlloc(&data->weight, 2*n*sizeof(float), cudaHostAllocDefault));
    checkCudaErrors(cudaHostAlloc(&data->buffer, 2*n*sizeof(float), cudaHostAllocDefault));
  }

#ifdef RANDOM_WEIGHTS
  for (j = 0; j < n; j++) {
    int idx = j+1;
    data->weight[j*2] = sin(idx) / ((float)idx);
    data->weight[j*2+1] = cos(idx) / ((float)idx);
    data->buffer[j*2] = 0.0;
    data->buffer[j*2+1] = 0.0;
  }
#else
  data->weight[0] = 1.0;
  data->weight[1] = 0.0;
  for (i = 1; i < 2*n; i++) {
    data->weight[i] = 0.0;
    data->buffer[i] = 0.0;
  }
#endif
}

void BeamFormWeights(int beam, float *weights)
{
  int i;
  for (i = 0; i < NUM_CHANNELS; i++)
  {
#ifdef RANDOM_WEIGHTS
    int idx = i+1;
    weights[2*i] = sin(idx) / (float)(beam+idx);
    weights[2*i+1] = cos(idx) / (float)(beam+idx);
#else
    if (i == beam) {
      weights[2*i] = 1;
      weights[2*i+1] = 0;
    } else {
      weights[2*i] = 0;
      weights[2*i+1] = 0;
    }
#endif
  }
}

void BeamForm(int beam, const float *weights, const float *input,
              float *output)
{
  /* 2*NUM_CHANNELS inputs and weights; 2 outputs. */
  float real_curr = 0;
  float imag_curr = 0;
  int i;
  for (i = 0; i < NUM_CHANNELS; i++)
  {
    real_curr += weights[2*i] * input[2*i] - weights[2*i+1] * input[2*i+1];
    imag_curr += weights[2*i] * input[2*i+1] + weights[2*i+1] * input[2*i];
  }
  output[0] = real_curr;
  output[1] = imag_curr;
}

void Magnitude(float *in, float *out, int n)
{
  int i;
  /* Should be 2n inputs, n outputs. */
  for (i = 0; i < n; i++)
    out[i] = sqrt(in[2*i]*in[2*i] + in[2*i+1]*in[2*i+1]);
}

void Detector(int beam, float *data, float *output)
{
  int sample;
  /* Should be exactly NUM_POST_DEC_2 samples. */
  for (sample = 0; sample < NUM_POST_DEC_2; sample++)
  {
    float outputVal;
    if (beam == TARGET_BEAM && sample == TARGET_SAMPLE) {
      if (data[sample] >= 0.1)
        outputVal = beam+1;
      else
        outputVal = 0;
    } else {
      if (data[sample] >= 0.1)
        outputVal = -(beam+1);
      else
        outputVal = 0;
    }
    outputVal = data[sample];
    output[sample]= outputVal;
  }
}


void BeamFirFilter(BeamFirData *data,
                   int input_length, int decimation_ratio,
                   float *in, float *out)
{
  /* Input must be exactly 2*decimation_ratio long; output must be
   * exactly 2 long. */
  //float real_curr = 0;
  //float imag_curr = 0;
  int i, j;
  int modPos;
  int len, mask, mask2;
  for(j = 0; j < input_length; j++){
    float real_curr = 0;
    float imag_curr = 0;
    len = data->len;
    mask = len - 1;
    mask2 = 2 * len - 1;
    modPos = 2*(len - 1 - data->pos);
    if(decimation_ratio != 1){
      data->buffer[modPos] = in[j * decimation_ratio * 2 ];
      data->buffer[modPos+1] = in[j * decimation_ratio * 2 + 1];
    }else{
      data->buffer[modPos] = in[j * 2 ];
      data->buffer[modPos+1] = in[j * 2 + 1];
    }

    /* Profiling says: this is the single inner loop that matters! */
    for (i = 0; i < 2*len; i+=2) {
      float rd = data->buffer[modPos];
      float id = data->buffer[modPos+1];
      float rw = data->weight[i];
      float iw = data->weight[i+1];
      float rci = rd * rw + id * iw;
      /* sign error?  this is consistent with StreamIt --dzm */
      float ici = id * rw + rd * iw;
      real_curr += rci;
      imag_curr += ici;
      modPos = (modPos + 2) & mask2;
    }
    data->pos = (data->pos + 1) & mask;
    out[j * 2] = real_curr;
    out[j * 2 + 1] = imag_curr;
  }
}

__global__ void BeamFirFilter_dev(int len,
                        int input_length, int decimation_ratio,
                        float *weight, float *buffer,
                        float *in, float *out)
{
  /* Input must be exactly 2*decimation_ratio long; output must be
   * exactly 2 long. */
  int tid = threadIdx.x + blockIdx.x *blockDim.x;
  int i, j;
  int modPos;
  int mask, mask2;

  //len = data->len;
  mask = len - 1;
  mask2 = 2 * len - 1;
  //for(k = 0; k < TD_NUM; k++){
  if(tid < TDD_NUM){
    for(j = 0; j < (input_length/TDD_NUM); j++){
      float real_curr = 0;
      float imag_curr = 0;
      //modPos = 2*(len - 1 - data->pos);
      modPos = 2*(len - 1 - ((j*TDD_NUM+tid) & mask));
      buffer[modPos] = in[(j*TDD_NUM+tid) * decimation_ratio * 2 ];
      buffer[modPos+1] = in[(j*TDD_NUM+tid) * decimation_ratio * 2 + 1];

      /* Profiling says: this is the single inner loop that matters! */
      for (i = 0; i < 2*len; i+=2) {
        float rd = buffer[modPos];
        float id = buffer[modPos+1];
        float rw = weight[i];
        float iw = weight[i+1];
        float rci = rd * rw + id * iw;
        /* sign error?  this is consistent with StreamIt --dzm */
        float ici = id * rw + rd * iw;
        real_curr += rci;
        imag_curr += ici;
        modPos = (modPos + 2) & mask2;
      }
      //data->pos = (data->pos + 1) & mask;
      out[(j*TDD_NUM+tid) * 2] = real_curr;
      out[(j*TDD_NUM+tid) * 2 + 1] = imag_curr;
    }
  }
}

