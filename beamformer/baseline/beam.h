#ifndef _GPU_BEAM_H_
#define _GPU_BEAM_H_

#define NUM_SAMPLES 4096
#define NUM_BEAMS 4
/* Bill writes: under current implementation, decimation ratios must
   divide NUM_SAMPLES. */
#define NUM_COARSE_TAPS 256
#define NUM_FINE_TAPS 256
#define COARSE_DECIMATION_RATIO 1
#define FINE_DECIMATION_RATIO 2
#define NUM_SEGMENTS 1
#define NUM_POST_DEC_1 (NUM_SAMPLES/COARSE_DECIMATION_RATIO)
#define NUM_POST_DEC_2 (NUM_POST_DEC_1/FINE_DECIMATION_RATIO)
#define MF_SIZE (NUM_SEGMENTS*NUM_POST_DEC_2)
#define PULSE_SIZE (NUM_POST_DEC_2/2)
#define PREDEC_PULSE_SIZE \
  (PULSE_SIZE*COARSE_DECIMATION_RATIO*FINE_DECIMATION_RATIO)
#define TARGET_BEAM (NUM_BEAMS/4)
#define TARGET_SAMPLE (NUM_SAMPLES/4)
#define TARGET_SAMPLE_POST_DEC \
   (TARGET_SAMPLE/COARSE_DECIMATION_RATIO/FINE_DECIMATION_RATIO)
#define D_OVER_LAMBDA 0.5
#define CFAR_THRESHOLD (0.95*D_OVER_LAMBDA*NUM_CHANNELS*0.5*PULSE_SIZE)
#define RANDOM_INPUTS 1

typedef struct 
{
  int len, count, pos;
  float *weight, *buffer;
}BeamFirData;

void BeamFirSetup(BeamFirData *data, int n);
void BeamFormWeights(int beam, float *weights, int TD_NUM);
void InputGenerate(int channel, float *inputs, int n);
void BeamForm(int beam, const float *weights, const float *input,
              float *output, int TD_NUM);
void Magnitude(float *in, float *out, int n);
void Detector(int beam, float *data, float *output);

void BeamFirFilter(BeamFirData *data,
                   int input_length, int decimation_ratio,
                   float *in, float *out);
__global__ void BeamFirFilter_dev(int len,
                        int input_length, int decimation_ratio,
                        float *weight, float *buffer,
                        float *in, float *out, int TD_NUM);
#endif
