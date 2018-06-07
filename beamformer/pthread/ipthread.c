/*
 * Copyright (c) 2003 David Maze
 *
 * Permission  is hereby  granted,  free  of  charge, to  any  person
 * obtaining a  copy of  this software  and  associated documentation
 * files   (the  "Software"),  to   deal  in  the   Software  without
 * restriction, including without  limitation the rights to use, copy,
 * modify, merge, publish,  distribute, sublicense, and/or sell copies
 * of  the Software,  and to  permit persons to  whom the  Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above  copyright notice  and this  permission notice  shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE  SOFTWARE IS  PROVIDED "AS IS",  WITHOUT WARRANTY OF  ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING  BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY,   FITNESS   FOR   A   PARTICULAR    PURPOSE    AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM,  DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,  OUT OF OR IN
 * CONNECTION  WITH THE SOFTWARE OR  THE USE OR OTHER  DEALINGS IN THE
 * SOFTWARE.  
 */

/*
 * beamformer.c: Standalone beam-former reference implementation
 * David Maze <dmaze@cag.lcs.mit.edu>
 * $Id: beamformer.c,v 1.5 2003/11/07 08:47:00 thies Exp $
 */

/* Modified by: Rodric M. Rabbah 06-03-04 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <pthread.h>
#include "beam.h"
/* 
This implementation is derived from the StreamIt implementation,
rather than from the PCA VSIPL-based implementation.  It is intended
to be easier to synchronize our reference implementation with our
StreamIt implementation, not have dependencies on extra libraries, and
generally be a fairer comparison with the StreamIt code (that is,
equivalent to our other benchmarks).

This version gets consistent output with the StreamIt version when
compiled with RANDOM_INPUTS and RANDOM_WEIGHTS.  The algorithm should
be equivalent to the SERIALIZED implementation.

FLAGS: the StreamIt TemplateBeamFormer can be built with COARSE
defined or not, with SERIALIZED defined or not, and with RANDOM_INPUTS
defined or not.  Defining COARSE changes the algorithm to add two
decimation stages; this presently isn't implemented.  SERIALIZED
changes where in the graph printing happens; in the non-SERIALIZED
version printing happens in the detector stage, but this doesn't have
a deterministic order.  RANDOM_INPUTS and RANDOM_WEIGHTS change
weights at several points in the code, where "RANDOM" is "something
the author made up" rather than "determined via rand()".  The default
flags are both RANDOM_WEIGHTS and RANDOM_INPUTS.

RANDOM_INPUTS and RANDOM_WEIGHTS are easy to do, so they are
implemented here as well.  We ignore SERIALIZED, though we do go
through the detectors in a deterministic order which should give the
right answer, and add a reordering stage equivalent to what SERIALZIED
adds to get results in the same order.  COARSE is left for a future
implementation.
*/

#define RANDOM_INPUTS

double my_timer()
{
struct timeval time;
double _ret_val_0;
gettimeofday(( & time), 0);
_ret_val_0=(time.tv_sec+(time.tv_usec/1000000.0));
return _ret_val_0;
}

struct BeamFirData
{
  int len, count, pos;
  float *weight, *buffer;
};

void InputGenerate(int channel, float *inputs, int n);
void BeamFirSetup(struct BeamFirData *data, int n);
void BeamFirFilter(struct BeamFirData *data,
                   int input_length, int decimation_ratio,
                   float *in, float *out);
void BeamFirFilter_omp(struct BeamFirData *data,
                   int input_length, int decimation_ratio,
                   float *in, float *out);

void BeamFormWeights(int beam, float *weights);
void BeamForm(int beam, const float *weights, const float *input,
              float *output);
void Magnitude(float *in, float *out, int n);
void Detector(int beam, float *data, float *output);

typedef struct
{
  struct BeamFirData a;
  int b, c;
  float **d, **e;
} parm;

void * worker(void *arg)
{
        parm           *p = (parm *) arg;
        BeamFirFilter_omp(&(p->a), p->b, p->c, *(p->d), *(p->e));
}


/* What are the counts here?  Deriving:
 *
 * Detector takes TARGET_SAMPLE_POST_DEC inputs.
 * Magnitude takes twice as many.
 * The various BeamFirFilters have internal circular buffers,
 *   so their counts are only affected by the decimation rates.
 *   The last BeamFirFilter has no decimation, so its input count
 *   is also 2*TARGET_SAMPLE_POST_DEC.
 * BeamForm takes NUM_CHANNELS times as many.
 *
 * So, coming out of the first for loop, the buffer should have
 * 2*TARGET_SAMPLE_POST_DEC*NUM_CHANNELS values.  This comes from
 * a roundrobin(2) joiner in StreamIt, for NUM_CHANNELS children.
 *
 * The second BeamFirFilter has 2*TARGET_SAMPLE_POST_DEC outputs.
 * The first BeamFirFilter has
 *   2*TARGET_SAMPLE_POST_DEC*FINE_DECIMATION_RATIO outputs.
 * The input generator produces 2*TARGET_SAMPLE values.
 */

int main(int argc, char *argv[])
{

  	volatile float result;
  	struct BeamFirData *coarse_fir_data;
  	struct BeamFirData *fine_fir_data;
  	struct BeamFirData *mf_fir_data;
  	float **inputs;
  	float **predec;
  	float **postdec;
  	float **predec_OpenMP;
  	float **postdec_OpenMP;
  	float **beam_weights;
  	float *beam_input;
  	float *beam_output;
  	float *beam_fir_output;
  	float *beam_fir_mag;
  	float **detector_out;

  	int num_thread[NUM_CHANNELS];
  	int num_size[NUM_CHANNELS];
  	FILE *f;

  	int i, j, k;

  	parm           *arg;
  	pthread_t      *threads;
  	pthread_attr_t  attrs;


  	pthread_attr_init(&attrs);
  	pthread_setconcurrency(16);
  	pthread_attr_setstacksize(&attrs, THREADSTACK);

  	threads = (pthread_t *) malloc(NUM_CHANNELS * sizeof(pthread_t));
  	arg=(parm *)malloc(sizeof(parm)*NUM_CHANNELS);

  	f = fopen("rand4.txt", "r");
  	for(i = 0; i < NUM_CHANNELS; i++)
    		fscanf(f, "%1d", &num_thread[i]);

  	fclose(f);

  	for(i = 0; i < NUM_CHANNELS; i++){
    		num_size[i] = (num_thread[i]*16)*(num_thread[i]*16);
    //printf("num_size:%d\n", num_size[i]);
  	}

  	inputs =  (float**)malloc(NUM_CHANNELS * sizeof(float *));
  	predec =  (float**)malloc(NUM_CHANNELS * sizeof(float *));
  	postdec =  (float**)malloc(NUM_CHANNELS * sizeof(float *));

  	predec_OpenMP =  (float**)malloc(NUM_CHANNELS * sizeof(float *));
  	postdec_OpenMP =  (float**)malloc(NUM_CHANNELS * sizeof(float *));

  	beam_weights =  (float**)malloc(NUM_BEAMS * sizeof(float *));
  	detector_out =  (float**)malloc(NUM_BEAMS * sizeof(float *));
  	coarse_fir_data =  (struct BeamFirData*)malloc(NUM_CHANNELS * sizeof(struct BeamFirData));
  	fine_fir_data =  (struct BeamFirData*)malloc(NUM_CHANNELS * sizeof(struct BeamFirData));
  	mf_fir_data =  (struct BeamFirData*)malloc(NUM_CHANNELS * sizeof(struct BeamFirData));


  	for(i = 0; i < NUM_CHANNELS; i++){
    		inputs[i] = (float*)malloc(2*num_size[i]*sizeof(float));
    		postdec[i] = (float*)malloc(2*num_size[i]*sizeof(float));
    		predec[i] = (float*)malloc(2*num_size[i]*sizeof(float));
    		postdec_OpenMP[i] = (float*)malloc(2*num_size[i]*sizeof(float));
    		predec_OpenMP[i] = (float*)malloc(2*num_size[i]*sizeof(float));

  	}

  	for (i = 0; i < NUM_CHANNELS; i++)
  	{
    		BeamFirSetup(&coarse_fir_data[i], num_size[i]);
    		BeamFirSetup(&fine_fir_data[i], num_size[i]);
  	}

  	/*** VERSABENCH START ***/
  	// Generate Input data  
  	for (i = 0; i < NUM_CHANNELS; i++) {
    		InputGenerate(i, inputs[i],
                  	num_size[i]);
  	}

  	double start_timer, end_timer;
  	start_timer = my_timer();

  	// Beam filter
  
  	for(k = 0; k < LOOP_NUM; k++){
    		for(i = 0; i < sub_task; i++){
        		arg[i].a = coarse_fir_data[k*sub_task+i];
        		arg[i].b = num_size[k*sub_task+i];
        		arg[i].c = COARSE_DECIMATION_RATIO;
			arg[i].d = &inputs[k*sub_task+i];
        		arg[i].e = &predec[k*sub_task+i];

        		pthread_create(&threads[i], &attrs, worker, (void *)(arg+i));

    		}

    		for (i = 0; i < sub_task; i++){
      			pthread_join(threads[i], NULL);
    		}

  	}

  	for(k = 0; k < LOOP_NUM; k++){
    		for(i = 0; i < sub_task; i++){
        		arg[i].a = fine_fir_data[k*sub_task+i];
        		arg[i].b = num_size[k*sub_task+i];
        		arg[i].c = FINE_DECIMATION_RATIO;
        		arg[i].d = &predec[k*sub_task+i];
        		arg[i].e = &postdec[k*sub_task+i];

	        	pthread_create(&threads[i], &attrs, worker, (void *)(arg+i));

    		}

    		for (i = 0; i < sub_task; i++){
      			pthread_join(threads[i], NULL);
    		}

  	}

  	end_timer = my_timer();
  	printf("Beamformer pthread Elapsed Time: %lf Sec.\n", end_timer - start_timer);

#if 0
  	start_timer = my_timer();
  	// Beam filter
    	for (i = 0; i < NUM_CHANNELS; i++) {
      		BeamFirFilter(&coarse_fir_data[i],
                  num_size[i], COARSE_DECIMATION_RATIO,
                  inputs[i],
                  predec_OpenMP[i]);

    	}

  	for (i = 0; i < NUM_CHANNELS; i++) {
    		BeamFirFilter(&fine_fir_data[i],
                  num_size[i], FINE_DECIMATION_RATIO,
                  predec_OpenMP[i],
                  postdec_OpenMP[i]);
  	}
  	end_timer = my_timer();
  	printf("The CPU Elapsed Time:%lf Sec.\n", end_timer - start_timer);


  	// verification
  
  	for(i = 0; i < 1; i++){
    		for(j = 0; j < 2*num_size[i]; j++){
      			if(postdec[i][j] != postdec_OpenMP[i][j]){
				printf("Error:%f, %f, %d, %d\n", postdec[i][j], postdec_OpenMP[i][j], i, j);
        			break;
      			}
    		}
  	}

#endif

  	/*** VERSABENCH END ***/

  	// Free memory
  	for(i = 0; i < NUM_CHANNELS; i++){
    		free(inputs[i]);
    		free(postdec[i]);
    		free(predec[i]);
    		free(postdec_OpenMP[i]);
    		free(predec_OpenMP[i]);

  	}
#if 0
  	for(i = 0; i < NUM_BEAMS; i++){
    		free(beam_weights[i]);
    		free(detector_out[i]);
  	}
#endif
  	//free(predec);
  	free(beam_input);
  	free(beam_output);
  	free(beam_fir_output);
  	free(beam_fir_mag);
  	free(inputs);
  	free(predec);
  	free(postdec);
  	free(predec_OpenMP);
  	free(postdec_OpenMP);
  	free(beam_weights);
  	free(detector_out);
  	free(coarse_fir_data);
  	free(fine_fir_data);
  	free(mf_fir_data);

  	free(arg);

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

void BeamFirSetup(struct BeamFirData *data, int n)
{
  int i, j;
  
  data->len = n;
  data->count = 0;
  data->pos = 0;
  data->weight = malloc(sizeof(float)*2*n);
  data->buffer = malloc(sizeof(float)*2*n);
  
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

void BeamFirFilter(struct BeamFirData *data,
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

void BeamFirFilter_omp(struct BeamFirData *data,
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
  float real_curr, imag_curr;
  float rd, id, rw, iw, rci, ici;

  for(j = 0; j < input_length; j++){
    real_curr = 0;
    imag_curr = 0;
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
      rd = data->buffer[modPos];
      id = data->buffer[modPos+1];
      rw = data->weight[i];
      iw = data->weight[i+1];
      rci = rd * rw + id * iw;
      /* sign error?  this is consistent with StreamIt --dzm */
      ici = id * rw + rd * iw;
      real_curr += rci;
      imag_curr += ici;
      modPos = (modPos + 2) & mask2;
    }
    data->pos = (data->pos + 1) & mask;
    out[j * 2] = real_curr;
    out[j * 2 + 1] = imag_curr;
  }
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

