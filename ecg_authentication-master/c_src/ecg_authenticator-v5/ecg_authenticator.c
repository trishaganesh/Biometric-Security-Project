/*

ECG Authenticator using Pan Tompkins
v5

Created by Renato Cordeiro <renato.silveiracordeiro@sjsu.edu>

This project reads an ECG from file and performs the following tasks:
1) Filters the signal
2) Detects R-Peak & perform segmentation
3) Detects Fiducial Points (P,Q,R,S,T and normalize them)
4) Runs a matching algorithm against a previoulsy generated set of templates and identify the user whose ECG template
    has the closest match (using Euclidian distances)


The steps 1 and 2 above are part of the Pan Tompkins algorithm. The implementation used in this project
was created by Rafael de Moura Moreira <rafaelmmoreira@gmail.com> and licensed under MIT License. The license
can be seen in the LICENSE file or in his repository:
ANSI-C implementation of Pan-Tompkins real-time QRS detection algorithm (https://github.com/rafaelmmoreira/PanTompkinsQRS)


The Matching algoritm uses Euclidian Distance between the feature vector (templates). That algorithm was inspired by Shen et al. paper
named 'Implementation of a one-lead ECG human identification system on a normal population' available at http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.877.7121&rep=rep1&type=pdf


*/

// Settings - Signal (specify those in the initialize_xxxxx() function)
int SIGNAL_SIZE_IN_SECS = -1;
int SAMPLING_RATE = -1;


// Settings - Pan-Tompkins (specify those in the initialize_xxxxx() function)
#define FILTER_DELAY_IN_SAMPLES 22
int BUFFER_SIZE = -1;   // The size of the buffers (in samples). Must fit more than 1.66 times an RR interval, which typically could be around 1 second.
int WINDOW_SIZE = -1;   // Integration WINDOW_SIZE, in samples, must be defined so that the window is ~150ms.


// Settings - Model
#define NUMBER_OF_FEATURES 10
int NUMBER_OF_USERS = -1;


// imports
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>



typedef enum {false, true} bool;

bool IS_VERBOSE = true;
bool DISABLE_PRINTS_FOR_SIMULATION = false;

// long int SIGNAL[SIGNAL_SIZE_IN_SECS * SAMPLING_RATE];
long int * SIGNAL;
int SIGNAL_LENGTH = -1;

// long int R_PEAKS[SIGNAL_SIZE_IN_SECS*10];
long int * R_PEAKS;
int R_PEAKS_LENGTH = 0;
int SAMPLE_COUNT = 0;

float ** LIST_TEMPLATES; // the database of templates. Code will compare the user template generated from the current ECG against each of the templates in LIST_TEMPLATES
int LIST_TEMPLATES_LENGTH;




void output(int v){
  // printf("%d\n", v);
  if (v) {
    R_PEAKS_LENGTH++;
    R_PEAKS[R_PEAKS_LENGTH] = SAMPLE_COUNT;
  }
  SAMPLE_COUNT++;
}

// reads the ECG signal from filename
bool step1_readSignal(char signal_filename[]) {
  if (IS_VERBOSE)
    printf("bool result = step1_readSignal: Reading Signal from file: %s\n", signal_filename);
  FILE *fin = fopen(signal_filename, "r");
  long int data = -99999;
  int i = 0;
  if (fin == NULL)
    return false;

  while (!feof(fin) && i < SIGNAL_SIZE_IN_SECS * SAMPLING_RATE) {
    fscanf(fin, "%li", &data);
    SIGNAL[i] = data;
    i++;
  }
  SIGNAL_LENGTH = i;
  if (IS_VERBOSE)
    printf("\nbool result = step1_readSignal: Finished reading Signal. Total number of samples = %d\n\n", SIGNAL_LENGTH);
  return true;
}

// filter the signal as specified by Pan-Tompkins
long int * step2_filtering(){
  if (IS_VERBOSE)
    printf("step2_filtering - starting...\n\n");
  long int * dcblock = (long int*) calloc (SIGNAL_LENGTH, sizeof(long int));
  long int * lowpass = (long int*) calloc (SIGNAL_LENGTH, sizeof(long int));
  long int * highpass = (long int*) calloc (SIGNAL_LENGTH, sizeof(long int));

  for(int i=0; i<SIGNAL_LENGTH; i++){
    // DC Block filter
    // This was not proposed on the original paper.
    // It is not necessary and can be removed if your sensor or database has no DC noise.
    if (i >= 1)
      dcblock[i] = SIGNAL[i] - SIGNAL[i-1] + 0.995*dcblock[i-1];
    else
      dcblock[i] = 0;

    // Low Pass filter
    // Implemented as proposed by the original paper:  y(nT) = 2y(nT - T) - y(nT - 2T) + x(nT) - 2x(nT - 6T) + x(nT - 12T)
    lowpass[i] = dcblock[i];
    if (i >= 1)
      lowpass[i] += 2*lowpass[i-1];
    if (i >= 2)
      lowpass[i] -= lowpass[i-2];
    if (i >= 6)
      lowpass[i] -= 2*dcblock[i-6];
    if (i >= 12)
      lowpass[i] += dcblock[i-12];
  // printf("%d\n", lowpass[i]);
    // High Pass filter
    // Implemented as proposed by the original paper:  y(nT) = 32x(nT - 16T) - [y(nT - T) + x(nT) - x(nT - 32T)]
    highpass[i] = -lowpass[i];
    if (i >= 1)
      highpass[i] -= highpass[i-1];
    if (i >= 16)
      highpass[i] += 32*lowpass[i-16];
    if (i >= 32)
      highpass[i] += lowpass[i-32];
    // printf("%d\n", highpass[i]);
  }
  free(dcblock);
  free(lowpass);
  if (IS_VERBOSE)
    printf("step2_filtering - ended\n\n");
  return highpass;
}

// identify R-peaks using Pan-Tompkins algorithm
void step3_segmenter(long int * signalFiltered){
  if (IS_VERBOSE)
    printf("step3_segmenter - starting...\n\n");

  long int * highpass = (long int*) calloc (BUFFER_SIZE, sizeof(long int));
  long int * derivative = (long int*) calloc (BUFFER_SIZE, sizeof(long int));
  long int * squared = (long int*) calloc (BUFFER_SIZE, sizeof(long int));
  long int * integral = (long int*) calloc (BUFFER_SIZE, sizeof(long int));
  long int * outputSignal = (long int*) calloc (BUFFER_SIZE, sizeof(long int));

  // for(int i=0; i<SIGNAL_LENGTH; i++){ printf("%d\n", signalFiltered[i]); }

  int rr1[8], rr2[8], rravg1, rravg2, rrlow = 0, rrhigh = 0, rrmiss = 0;
  int i, j, lastQRS = 0, lastSlope = 0;
  int peak_i = 0, peak_f = 0, threshold_i1 = 0, threshold_i2 = 0, threshold_f1 = 0, threshold_f2 = 0, spk_i = 0, spk_f = 0, npk_i = 0, npk_f = 0;
  bool qrs, regular = true, prevRegular;
  int current; // signal buffers index. If the buffers still aren't completely filled, it shows the last filled position. Once the buffers are full, it'll always show the last position, and new samples will make the buffers shift, discarding the oldest sample and storing the newest one on the last position.


  // Initializing the RR averages
  for (i = 0; i < 8; i++) {
    rr1[i] = 0;
    rr2[i] = 0;
  }


  int FS = SAMPLING_RATE;
  int DELAY = FILTER_DELAY_IN_SAMPLES;

  int sample = 0;


  for(sample = 0; sample < SIGNAL_LENGTH; sample++){
    // Test if the buffers are full.
    // If they are, shift them, discarding the oldest sample and adding the new one at the end.
    // Else, just put the newest sample in the next free position.
    // Update 'current' so that the program knows where's the newest sample.
    if (sample >= BUFFER_SIZE) {
      for (i = 0; i < BUFFER_SIZE - 1; i++) {
        highpass[i] = highpass[i+1];
        derivative[i] = derivative[i+1];
        squared[i] = squared[i+1];
        integral[i] = integral[i+1];
        outputSignal[i] = outputSignal[i+1];
      }
      current = BUFFER_SIZE - 1;
    } else {
      current = sample;
    }
    highpass[current] = signalFiltered[sample];



    // -------------------------------------
    // -- DERIVATIVE


    // Derivative filter
    // This is an alternative implementation, the central difference method.
    // f'(a) = [f(a+h) - f(a-h)]/2h
    // The original formula used by Pan-Tompkins was:
    // y(nT) = (1/8T)[-x(nT - 2T) - 2x(nT - T) + 2x(nT + T) + x(nT + 2T)]
    derivative[current] = highpass[current];
    if (current > 0)
      derivative[current] -= highpass[current-1];


    // -------------------------------------
    // -- SQUARING


    // This just squares the derivative, to get rid of negative values and emphasize high frequencies.
    // y(nT) = [x(nT)]^2.
    squared[current] = derivative[current]*derivative[current];




    // -------------------------------------
    // -- INTEGRATION


    // Moving-Window Integration
    // Implemented as proposed by the original paper.
    // y(nT) = (1/N)[x(nT - (N - 1)T) + x(nT - (N - 2)T) + ... x(nT)]

    integral[current] = 0;
    for (i = 0; i < WINDOW_SIZE; i++)
    {
      if (current >= (int)i){
         integral[current] += highpass[current - i];
         //printf("pos\n");
        }
      else
        break;
    }
    integral[current] /= (int)i;





    // -------------------------------------
    // -- PEAK DETECTION

    qrs = false;

    // If the current signal is above one of the thresholds (integral or filtered signal), it's a peak candidate.
    if (integral[current] >= threshold_i1 || highpass[current] >= threshold_f1)
    {
        peak_i = integral[current];
        peak_f = highpass[current];
    }

    // If both the integral and the signal are above their thresholds, they're probably signal peaks.
    if ((integral[current] >= threshold_i1) && (highpass[current] >= threshold_f1))
    {
      // There's a 200ms latency. If the new peak respects this condition, we can keep testing.
      if (sample > lastQRS + 0.2*FS)
      {
          // If it respects the 200ms latency, but it doesn't respect the 360ms latency, we check the slope.
          if (sample <= lastQRS + 0.36*FS)
          {
              if (squared[current] < (int)(lastSlope/2))
                      {
                          qrs = false;
                      }

                      else
                      {
                          spk_i = 0.125*peak_i + 0.875*spk_i;
                          threshold_i1 = npk_i + 0.25*(spk_i - npk_i);
                          threshold_i2 = 0.5*threshold_i1;

                          spk_f = 0.125*peak_f + 0.875*spk_f;
                          threshold_f1 = npk_f + 0.25*(spk_f - npk_f);
                          threshold_f2 = 0.5*threshold_f1;

                          lastSlope = squared[current];
                          qrs = true;
                      }
          }
          // If it was above both thresholds and respects both latency periods, it certainly is a R peak.
          else
          {
              if (squared[current] > (int)(lastSlope/2))
                      {
                          spk_i = 0.125*peak_i + 0.875*spk_i;
                          threshold_i1 = npk_i + 0.25*(spk_i - npk_i);
                          threshold_i2 = 0.5*threshold_i1;

                          spk_f = 0.125*peak_f + 0.875*spk_f;
                          threshold_f1 = npk_f + 0.25*(spk_f - npk_f);
                          threshold_f2 = 0.5*threshold_f1;

                          lastSlope = squared[current];
                          qrs = true;
                      }
          }
      }
      // If the new peak doesn't respect the 200ms latency, it's noise. Update thresholds and move on to the next sample.
      else
            {
                peak_i = integral[current];
                npk_i = 0.125*peak_i + 0.875*npk_i;
                threshold_i1 = npk_i + 0.25*(spk_i - npk_i);
                threshold_i2 = 0.5*threshold_i1;
                peak_f = highpass[current];
                npk_f = 0.125*peak_f + 0.875*npk_f;
                threshold_f1 = npk_f + 0.25*(spk_f - npk_f);
                threshold_f2 = 0.5*threshold_f1;
                qrs = false;
                outputSignal[current] = qrs;
                if (sample > DELAY + BUFFER_SIZE){
                          output(outputSignal[0]);
                        }
                continue;
            }

    }

    // If a R-peak was detected, the RR-averages must be updated.
    if (qrs)
    {

      // Add the newest RR-interval to the buffer and get the new average.
      rravg1 = 0;
      for (i = 0; i < 7; i++)
      {
        rr1[i] = rr1[i+1];
        rravg1 += rr1[i];
      }
      rr1[7] = sample - lastQRS;
      lastQRS = sample;
      rravg1 += rr1[7];
      rravg1 *= 0.125;

      // If the newly-discovered RR-average is normal, add it to the "normal" buffer and get the new "normal" average.
      // Update the "normal" beat parameters.
      if ( (rr1[7] >= rrlow) && (rr1[7] <= rrhigh) )
      {
        rravg2 = 0;
        for (i = 0; i < 7; i++)
        {
          rr2[i] = rr2[i+1];
          rravg2 += rr2[i];
        }
        rr2[7] = rr1[7];
        rravg2 += rr2[7];
        rravg2 *= 0.125;
        rrlow = 0.92*rravg2;
        rrhigh = 1.16*rravg2;
        rrmiss = 1.66*rravg2;
      }

      prevRegular = regular;
      if (rravg1 == rravg2)
      {
        regular = true;
      }
      // If the beat had been normal but turned odd, change the thresholds.
      else
      {
        regular = false;
        if (prevRegular)
        {
          threshold_i1 /= 2;
          threshold_f1 /= 2;
        }
      }
    }
    // If no R-peak was detected, it's important to check how long it's been since the last detection.
    else
    {
        // If no R-peak was detected for too long, use the lighter thresholds and do a back search.
      // However, the back search must respect the 200ms limit.
      if ((sample - lastQRS > (long unsigned int)rrmiss) && (sample > lastQRS + 0.2*FS))
      {
        for (i = current - (sample - lastQRS) + 0.2*FS; i < (long unsigned int)current; i++)
        {
          if ( (integral[i] > threshold_i2) && (highpass[i] > threshold_f2))
          {
            peak_i = integral[i];
            peak_f = highpass[i];
            spk_i = 0.25*peak_i+ 0.75*spk_i;
            spk_f = 0.25*peak_f + 0.75*spk_f;
            threshold_i1 = npk_i + 0.25*(spk_i - npk_i);
            threshold_i2 = 0.5*threshold_i1;
            lastSlope = squared[i];
            threshold_f1 = npk_f + 0.25*(spk_f - npk_f);
            threshold_f2 = 0.5*threshold_f1;
            // If a signal peak was detected on the back search, the RR attributes must be updated.
            // This is the same thing done when a peak is detected on the first try.
            //RR Average 1
            rravg1 = 0;
            for (j = 0; j < 7; j++)
            {
              rr1[j] = rr1[j+1];
              rravg1 += rr1[j];
            }
            rr1[7] = sample - i - lastQRS;
            lastQRS = sample - i;
            if (((current-lastQRS) < BUFFER_SIZE) && ((current-lastQRS) >= 0))
              outputSignal[current-lastQRS] = true;
            rravg1 += rr1[7];
            rravg1 *= 0.125;

            //RR Average 2
            if ( (rr1[7] >= rrlow) && (rr1[7] <= rrhigh) )
            {
              rravg2 = 0;
              for (i = 0; i < 7; i++)
              {
                rr2[i] = rr2[i+1];
                rravg2 += rr2[i];
              }
              rr2[7] = rr1[7];
              rravg2 += rr2[7];
              rravg2 *= 0.125;
              rrlow = 0.92*rravg2;
              rrhigh = 1.16*rravg2;
              rrmiss = 1.66*rravg2;
            }

            prevRegular = regular;
            if (rravg1 == rravg2)
            {
              regular = true;
            }
            else
            {
              regular = false;
              if (prevRegular)
              {
                threshold_i1 /= 2;
                threshold_f1 /= 2;
              }
            }

            break;
          }
        }
      }
      // Definitely no signal peak was detected.
      if (!qrs)
      {
        // If some kind of peak had been detected, then it's certainly a noise peak. Thresholds must be updated accordinly.
        if ((integral[current] >= threshold_i1) || (highpass[current] >= threshold_f1))
        {
          peak_i = integral[current];
          npk_i = 0.125*peak_i + 0.875*npk_i;
          threshold_i1 = npk_i + 0.25*(spk_i - npk_i);
          threshold_i2 = 0.5*threshold_i1;
          peak_f = highpass[current];
          npk_f = 0.125*peak_f + 0.875*npk_f;
          threshold_f1 = npk_f + 0.25*(spk_f - npk_f);
          threshold_f2 = 0.5*threshold_f1;
        }
      }
    }
    // The current implementation outputs '0' for every sample where no peak was detected,
    // and '1' for every sample where a peak was detected. It should be changed to fit
    // the desired application.
    // The 'if' accounts for the delay introduced by the filters: we only start outputting after the delay.
    // However, it updates a few samples back from the buffer. The reason is that if we update the detection
    // for the current sample, we might miss a peak that could've been found later by backsearching using
    // lighter thresholds. The final waveform output does match the original signal, though.
    outputSignal[current] = qrs;
    if (sample > DELAY + BUFFER_SIZE){
      output(outputSignal[0]);
      }

  }

  // Output the last remaining samples on the buffer
  for (i = 1; i < BUFFER_SIZE; i++)
    output(outputSignal[i]);

  free(highpass);
  free(derivative);
  free(squared);
  free(integral);
  free(outputSignal);

  if (IS_VERBOSE)
    printf("step3_segmenter - ended\n\n");

}


// loops through the cardiac cycles identified by step3_segmenter, tries to correct the fiducial points and generate the features (normalized distanced and amplitudes)
float * step4_features_extraction(){
    if (IS_VERBOSE)
      printf("step4_features_extraction - starting...\n\n");

    int numTemplates = 0;
    float snP_x=0, snP_y=0, snQ_x=0, snQ_y=0, snR_x=0, snR_y=0, snS_x=0, snS_y=0, snT_x=0, snT_y=0;

    for (int i=0; i<R_PEAKS_LENGTH;i++){
        if (R_PEAKS[i] < 2 * SAMPLING_RATE) // ignoring the first 2 seconds
          continue;

        int R_x = R_PEAKS[i];


        //improving R_x
        int tentativeRx = R_x;
        int c;
        for(c=R_x; SIGNAL[c] >= SIGNAL[tentativeRx]; c++){
          tentativeRx = c;
        }
        for(c=R_x; SIGNAL[c] >= SIGNAL[tentativeRx]; c--){
          tentativeRx = c;
        }
        R_x = tentativeRx;


        int start = R_x - 0.2 * SAMPLING_RATE;
        int end = R_x + 0.4 * SAMPLING_RATE;


        // finding P, Q
        int Q_x = R_x - 0.150*SAMPLING_RATE;
        int P_x = -1;
        for (int c = Q_x; c< R_x; c++){
            if (SIGNAL[c] < SIGNAL[Q_x])
                Q_x = c;
        }
        if (Q_x == -1){
          if (IS_VERBOSE)
            printf("Q point not found!");
        } else {
          P_x = Q_x-1;
          for (int c = Q_x-1; c>= start; c--){
            if (SIGNAL[c] > SIGNAL[P_x])
              P_x = c;
          }
        }


        // finding T (highest peak after R)
        int T_x = R_x + 0.150*SAMPLING_RATE; // S-T interval varies from 150ms to 0.005s
        for (int c = T_x; c<= end; c++){
          if (SIGNAL[c] > SIGNAL[T_x]){
            T_x = c;
          }
        }

        // finding S  (must be between R and T)
        int S_x = R_x+1;
        for (int c = S_x; c<= T_x; c++){
            if (SIGNAL[c] < SIGNAL[S_x])
                S_x = c;
        }


        if (IS_VERBOSE)
          printf("Cycle #%d: P_x=%d, Q_x=%d, R_x=%d, S_x=%d, T_x=%d, start=%d, end=%d\n", i, P_x, Q_x, R_x, S_x, T_x, start, end);
        // if (i==21 && WINDOW_SIZE == 10 && BUFFER_SIZE == 51){
        //   printf("-------------------------\n\n");
        //   for(int z=0; z<SIGNAL_LENGTH; z++)
        //     printf("%d,%li\n", z, SIGNAL[z]);
        // printf("\n\n-------------------------\n\n");
        // }

        if (P_x > 0 && Q_x > 0 && R_x > 0 && S_x > 0 && T_x > 0){ //cycle has all points identified
          int P_y = SIGNAL[P_x], Q_y = SIGNAL[Q_x], R_y = SIGNAL[R_x], S_y = SIGNAL[S_x], T_y = SIGNAL[T_x];
          // printf("Cycle #%d: P_y=%d, Q_y=%d, R_y=%d, S_y=%d, T_y=%d, start=%d, end=%d\n", i, P_y, Q_y, R_y, S_y, T_y, start, end);

          // normalizing values
          float nP_x = 1.0 * (P_x - start) / (end - start);
          float nQ_x = 1.0 * (Q_x - start) / (end - start);
          float nR_x = 1.0 * (R_x - start) / (end - start);
          float nS_x = 1.0 * (S_x - start) / (end - start);
          float nT_x = 1.0 * (T_x - start) / (end - start);

          int max =  R_y;
          if (P_y > max) max =  P_y;
          if (T_y > max) max =  T_y;

          int min = Q_y;
          if (S_y < min)
            min = S_y;

          float nP_y = 1.0 * (P_y - min) / (max - min);
          float nQ_y = 1.0 * (Q_y - min) / (max - min);
          float nR_y = 1.0 * (R_y - min) / (max - min);
          float nS_y = 1.0 * (S_y - min) / (max - min);
          float nT_y = 1.0 * (T_y - min) / (max - min);

          // float tmplt[10] =  { nP_x, nQ_x, nR_x, nS_x, nT_x, nP_y, nQ_y, nR_y, nS_y, nT_y };
          // printf("Template= (%f, %f, %f, %f, %f, %f, %f, %f, %f, %f)\n\n", nP_x, nQ_x, nR_x, nS_x, nT_x, nP_y, nQ_y, nR_y, nS_y, nT_y);
          numTemplates++;
          snP_x += nP_x;
          snP_y += nP_y;
          snQ_x += nQ_x;
          snQ_y += nQ_y;
          snR_x += nR_x;
          snR_y += nR_y;
          snS_x += nS_x;
          snS_y += nS_y;
          snT_x += nT_x;
          snT_y += nT_y;
        }
    }

    float fP_x = snP_x / numTemplates;
    float fP_y = snP_y / numTemplates;
    float fQ_x = snQ_x / numTemplates;
    float fQ_y = snQ_y / numTemplates;
    float fR_x = snR_x / numTemplates;
    float fR_y = snR_y / numTemplates;
    float fS_x = snS_x / numTemplates;
    float fS_y = snS_y / numTemplates;
    float fT_x = snT_x / numTemplates;
    float fT_y = snT_y / numTemplates;

    if (IS_VERBOSE)
      printf("User Template= {%f, %f, %f, %f, %f, %f, %f, %f, %f, %f}\n\n", fP_x, fQ_x, fR_x, fS_x, fT_x, fP_y, fQ_y, fR_y, fS_y, fT_y);

    float * userTemplate = (float*) calloc (10, sizeof(float));
    userTemplate[0] = fP_x;
    userTemplate[1] = fQ_x;
    userTemplate[2] = fR_x;
    userTemplate[3] = fS_x;
    userTemplate[4] = fT_x;
    userTemplate[5] = fP_y;
    userTemplate[6] = fQ_y;
    userTemplate[7] = fR_y;
    userTemplate[8] = fS_y;
    userTemplate[9] = fT_y;

    if (IS_VERBOSE)
      printf("step4_features_extraction - ended\n\n");

    return userTemplate;
}

// calculates the euclidian distance between two vectors
double euclidian_distance(float *A, float *B, unsigned int size){
    double s = 0;
    for(int k=0; k<size;k++){
      s = s + ( (A[k] - B[k]) * (A[k] - B[k]));
    }
    return sqrt((double) s);
}

// receives an user template and calculates the euclidian distance against all previously registered templates. Returns the id that has the shortest distance to the template provided
int step5_matching(float * userTemplate){
  if (IS_VERBOSE)
    printf("step5_matching - starting...\n\n");
  int userId = -1;
  int prevUserId =-1;
  double preMinDist = 99999;
  double minDist = 99999;

  for(int i=0; i<LIST_TEMPLATES_LENGTH; i++){
    // printf("Testing userId %d - ", i);
    // fflush(stdout);
    float * t = LIST_TEMPLATES[i];
    double dist = euclidian_distance(t, userTemplate, 10);
    if (dist < minDist) {
      preMinDist = minDist;
      prevUserId = userId;
      minDist = dist;
      userId = i;
    }
  }
  if (IS_VERBOSE){
    printf("Closest to Template is userId %d with dist=%f\n", userId, (float) minDist);
    printf("2nd Closest to Template is userId %d with dist=%f\n", prevUserId, (float) preMinDist);
    printf("\nstep5_matching - ended\n\n");

  }

  return userId;
}


// - - - - - - - - - - - - - - - - - -
// TEMPLATES PRE-REGISTERED (The code below can be generated using the enrollAllUsers() function. Replace nan values with '9')


// WINDOWS SIZE = 12 (Accuracy=0.815789)
// float template_user[40][10]={
//     {0.071111,0.297531,0.333333,0.518025,0.833333,0.260203,0.028735,1.000000,0.031319,0.364076}, // user id=0
//     {0.191698,0.295975,0.333333,0.468931,0.780629,0.569774,0.425093,0.996799,0.000000,0.308637}, // user id=1
//     {0.258571,0.303333,0.333333,0.562143,0.804524,0.681595,0.552836,0.895531,0.007703,0.600797}, // user id=2
//     {0.190388,0.303411,0.333333,0.492713,0.795349,0.592605,0.318547,1.000000,0.005303,0.452577}, // user id=3
//     {0.012000,0.289600,0.333333,0.373600,0.817600,0.321884,0.156397,1.000000,0.000000,0.375931}, // user id=4
//     {0.183333,0.300794,0.333333,0.529048,0.814444,0.495744,0.341299,0.999379,0.034514,0.361231}, // user id=5
//     {0.127160,0.299506,0.333333,0.485926,0.841975,0.145699,0.000000,1.000000,0.051236,0.259488}, // user id=6
//     {0.180513,0.316752,0.333333,0.627180,0.837436,0.842005,0.401104,0.909197,0.000000,0.552843}, // user id=7
//     {0.207527,0.300645,0.333333,0.535699,0.868602,0.481871,0.319570,1.000000,0.010403,0.322608}, // user id=8
//     {0.235519,0.309836,0.333333,0.482404,0.784699,0.610838,0.304318,0.994018,0.003699,0.550085}, // user id=9
//     {0.211078,0.298333,0.333333,0.553725,0.805000,0.451599,0.369539,1.000000,0.003113,0.324312}, // user id=10
//     {0.222414,0.305287,0.333333,0.521379,0.772529,0.555190,0.382296,1.000000,0.001187,0.424516}, // user id=11
//     {0.151257,0.283060,0.333333,0.495519,0.768743,0.416067,0.315878,0.998717,0.000641,0.378193}, // user id=12
//     {0.206667,0.282941,0.333333,0.542353,0.751961,0.516527,0.447858,1.000000,0.000000,0.361505}, // user id=13
//     {0.214023,0.309080,0.333333,0.485058,0.750920,0.625453,0.329673,1.000000,0.000046,0.583841}, // user id=14
//     {0.193333,0.307483,0.333333,0.513334,0.768299,0.588821,0.438055,0.991011,0.000000,0.400162}, // user id=15
//     {0.043077,0.287692,0.333333,0.371282,0.793333,0.090249,0.003037,1.000000,0.016644,0.300839}, // user id=16
//     {0.117963,0.281111,0.333333,0.450926,0.791111,0.266481,0.130238,1.000000,0.069407,0.336968}, // user id=17
//     {0.090417,0.285000,0.333333,0.476667,0.753542,0.200953,0.031780,1.000000,0.015732,0.263555}, // user id=18
//     {0.102286,0.277333,0.333333,0.369333,0.759810,0.191528,0.071624,1.000000,0.000000,0.365236}, // user id=19
//     {0.149412,0.288627,0.333333,0.398431,0.779216,0.331972,0.241355,1.000000,0.000000,0.339820}, // user id=20
//     {0.273333,0.313333,0.333333,0.398667,0.690667,0.511914,0.137045,0.447823,0.049199,0.862216}, // user id=21
//     {0.112874,0.272184,0.333333,0.473793,0.773793,0.244482,0.031837,1.000000,0.140708,0.473590}, // user id=22
//     {0.175455,0.290606,0.333333,0.448788,0.854242,0.572148,0.458866,0.972409,0.000000,0.461530}, // user id=23
//     {0.220800,0.228267,0.333333,0.597333,0.900267,0.443158,0.428952,1.000000,0.000000,0.416859}, // user id=24
//     {0.040392,0.281569,0.333333,0.448627,0.853333,0.285021,0.096452,1.000000,0.040358,0.353382}, // user id=25
//     {0.107778,0.290000,0.333333,0.438519,0.843704,0.125619,0.002309,1.000000,0.010945,0.263148}, // user id=26
//     {9.000000,9.000000,9.000000,9.000000,9.000000,9.000000,9.000000,9.000000,9.000000,9.000000}, // user id=27
//     {0.144000,0.274667,0.333333,0.411333,0.794667,0.512834,0.339242,0.960108,0.023818,0.619097}, // user id=28
//     {0.079259,0.275556,0.333333,0.443704,0.796296,0.107062,0.002625,1.000000,0.008372,0.275146}, // user id=29
//     {0.117500,0.265000,0.333333,0.384167,0.772500,0.318292,0.236112,1.000000,0.000000,0.566865}, // user id=30
//     {9.000000,9.000000,9.000000,9.000000,9.000000,9.000000,9.000000,9.000000,9.000000,9.000000}, // user id=31
//     {0.113333,0.202333,0.333333,0.475000,0.681000,0.211532,0.171326,1.000000,0.000000,0.213414}, // user id=32
//     {0.261026,0.312308,0.333333,0.544615,0.856410,0.803433,0.721683,0.986457,0.000000,0.360646}, // user id=33
//     {0.070303,0.274546,0.333333,0.451515,0.744546,0.144100,0.003792,1.000000,0.006319,0.216976}, // user id=34
//     {0.081667,0.282333,0.333333,0.373333,0.747333,0.203879,0.082972,1.000000,0.000000,0.276874}, // user id=35
//     {0.053333,0.285833,0.333333,0.379167,0.809167,0.247115,0.000000,1.000000,0.143546,0.427918}, // user id=36
//     {0.088533,0.290933,0.333333,0.430933,0.780267,0.104127,0.011549,1.000000,0.004708,0.285054}, // user id=37
//     {0.100000,0.282222,0.333333,0.464444,0.748889,0.174059,0.005350,1.000000,0.027336,0.280118}, // user id=38
//     {0.134872,0.289231,0.333333,0.380000,0.775385,0.171898,0.037190,1.000000,0.000000,0.371431}  // user id=39
// };


// WINDOW_SIZE=13 - BUFFER_SIZE=520
float template_user[90][10]={
    {0.083651,0.281587,0.333333,0.370476,0.704444,0.253865,0.056415,1.000000,0.002344,0.465187}, // user id=0 (ecgid/person_01_rec_1_ecg.csv)
    {0.142540,0.272857,0.333333,0.376032,0.740952,0.591501,0.227614,0.975105,0.000000,0.832224}, // user id=1 (ecgid/person_02_rec_1_ecg.csv)
    {0.095625,0.258750,0.333333,0.447500,0.779792,0.449932,0.000964,0.986979,0.035104,0.606591}, // user id=2 (ecgid/person_03_rec_1_ecg.csv)
    {0.115208,0.181250,0.333333,0.478750,0.797083,0.491855,0.015824,1.000000,0.011652,0.649632}, // user id=3 (ecgid/person_04_rec_1_ecg.csv)
    {0.107436,0.166026,0.333333,0.509872,0.764231,0.689760,0.266863,0.912894,0.017919,0.861909}, // user id=4 (ecgid/person_05_rec_1_ecg.csv)
    {0.083077,0.156838,0.333333,0.506838,0.754701,0.730839,0.073274,0.882075,0.012962,0.901468}, // user id=5 (ecgid/person_06_rec_1_ecg.csv)
    {0.055965,0.213158,0.333333,0.468421,0.772982,0.564555,0.020303,0.985585,0.065653,0.779779}, // user id=6 (ecgid/person_07_rec_1_ecg.csv)
    {0.139667,0.250833,0.333333,0.452833,0.721667,0.449828,0.005464,1.000000,0.036171,0.574080}, // user id=7 (ecgid/person_08_rec_1_ecg.csv)
    {0.069885,0.193563,0.333333,0.474713,0.738621,0.702248,0.067716,0.910485,0.020190,0.853911}, // user id=8 (ecgid/person_09_rec_1_ecg.csv)
    {0.142879,0.228636,0.333333,0.387424,0.696212,0.521491,0.215815,0.987634,0.000000,0.743419}, // user id=9 (ecgid/person_10_rec_1_ecg.csv)
    {0.072727,0.174924,0.333333,0.537197,0.854545,0.751153,0.162381,0.801032,0.015300,0.829441}, // user id=10 (ecgid/person_11_rec_1_ecg.csv)
    {0.112857,0.274444,0.333333,0.487619,0.720952,0.280613,0.005150,1.000000,0.036518,0.327060}, // user id=11 (ecgid/person_12_rec_1_ecg.csv)
    {0.075208,0.105625,0.333333,0.638958,0.738958,0.720347,0.033513,1.000000,0.034740,0.741035}, // user id=12 (ecgid/person_13_rec_1_ecg.csv)
    {0.084314,0.156863,0.333333,0.540980,0.773137,0.849637,0.045651,0.881737,0.142826,0.928612}, // user id=13 (ecgid/person_14_rec_1_ecg.csv)
    {0.101453,0.168034,0.333333,0.570940,0.816068,0.782684,0.051173,0.931109,0.016929,0.894334}, // user id=14 (ecgid/person_15_rec_1_ecg.csv)
    {0.139298,0.242281,0.333333,0.433333,0.707719,0.453023,0.019736,1.000000,0.016849,0.650420}, // user id=15 (ecgid/person_16_rec_1_ecg.csv)
    {0.141667,0.275417,0.333333,0.483125,0.774375,0.348158,0.007721,1.000000,0.016579,0.382547}, // user id=16 (ecgid/person_17_rec_1_ecg.csv)
    {0.059048,0.210238,0.333333,0.444762,0.725238,0.501227,0.028792,1.000000,0.015904,0.658447}, // user id=17 (ecgid/person_18_rec_1_ecg.csv)
    {0.134127,0.220000,0.333333,0.474127,0.704762,0.556249,0.022907,0.996177,0.018309,0.640852}, // user id=18 (ecgid/person_19_rec_1_ecg.csv)
    {0.060741,0.130278,0.333333,0.653519,0.817685,0.908317,0.036471,0.838475,0.062615,0.926337}, // user id=19 (ecgid/person_20_rec_1_ecg.csv)
    {0.069425,0.160920,0.333333,0.502414,0.814483,0.665313,0.053639,0.857843,0.088214,0.919472}, // user id=20 (ecgid/person_21_rec_1_ecg.csv)
    {0.096543,0.176420,0.333333,0.547654,0.746420,0.885374,0.062998,0.851199,0.260892,0.970567}, // user id=21 (ecgid/person_22_rec_1_ecg.csv)
    {0.131410,0.209487,0.333333,0.472180,0.774487,0.687548,0.314074,0.941339,0.019767,0.784370}, // user id=22 (ecgid/person_23_rec_1_ecg.csv)
    {0.089487,0.277436,0.333333,0.465641,0.746410,0.511571,0.004442,1.000000,0.049289,0.586354}, // user id=23 (ecgid/person_24_rec_1_ecg.csv)
    {0.113509,0.277018,0.333333,0.416491,0.747193,0.471137,0.026170,1.000000,0.032863,0.567805}, // user id=24 (ecgid/person_25_rec_1_ecg.csv)
    {0.106167,0.236000,0.333333,0.390167,0.725500,0.222446,0.012844,1.000000,0.009073,0.319552}, // user id=25 (ecgid/person_26_rec_1_ecg.csv)
    {0.154697,0.246061,0.333333,0.361667,0.702424,0.387337,0.130159,1.000000,0.000000,0.599886}, // user id=26 (ecgid/person_27_rec_1_ecg.csv)
    {0.139848,0.265758,0.333333,0.371667,0.706061,0.420752,0.063272,0.999040,0.010781,0.730907}, // user id=27 (ecgid/person_28_rec_1_ecg.csv)
    {0.123333,0.222917,0.333333,0.457917,0.736111,0.197867,0.018106,1.000000,0.011322,0.362068}, // user id=28 (ecgid/person_29_rec_1_ecg.csv)
    {0.118333,0.276167,0.333333,0.391667,0.678667,0.173529,0.000000,1.000000,0.082133,0.321224}, // user id=29 (ecgid/person_30_rec_1_ecg.csv)
    {0.117361,0.275278,0.333333,0.369722,0.690000,0.293287,0.132657,1.000000,0.000000,0.400215}, // user id=30 (ecgid/person_31_rec_1_ecg.csv)
    {0.059167,0.228958,0.333333,0.433542,0.771667,0.288781,0.009255,1.000000,0.039148,0.557826}, // user id=31 (ecgid/person_32_rec_1_ecg.csv)
    {0.117576,0.272576,0.333333,0.372121,0.732121,0.453184,0.190272,1.000000,0.000000,0.549434}, // user id=32 (ecgid/person_33_rec_1_ecg.csv)
    {0.110400,0.285733,0.333333,0.484933,0.716133,0.357000,0.014651,1.000000,0.029153,0.399618}, // user id=33 (ecgid/person_34_rec_1_ecg.csv)
    {0.113636,0.292273,0.333333,0.424848,0.675909,0.265669,0.000000,1.000000,0.086476,0.485617}, // user id=34 (ecgid/person_35_rec_1_ecg.csv)
    {0.129111,0.201333,0.333333,0.398444,0.798667,0.610667,0.391739,0.861463,0.039324,0.934257}, // user id=35 (ecgid/person_36_rec_1_ecg.csv)
    {0.130702,0.278947,0.333333,0.488947,0.781228,0.288551,0.002304,1.000000,0.048600,0.447999}, // user id=36 (ecgid/person_37_rec_1_ecg.csv)
    {0.062424,0.198182,0.333333,0.406061,0.754545,0.545461,0.048464,0.903163,0.031975,0.953561}, // user id=37 (ecgid/person_38_rec_1_ecg.csv)
    {0.116042,0.260833,0.333333,0.397500,0.752917,0.239495,0.019980,1.000000,0.009038,0.484188}, // user id=38 (ecgid/person_39_rec_1_ecg.csv)
    {0.099545,0.208939,0.333333,0.452424,0.705303,0.366764,0.037443,1.000000,0.041399,0.541036}, // user id=39 (ecgid/person_40_rec_1_ecg.csv)
    {0.147917,0.288542,0.333333,0.519167,0.791250,0.342535,0.000000,1.000000,0.054587,0.423326}, // user id=40 (ecgid/person_41_rec_1_ecg.csv)
    {0.151528,0.270694,0.333333,0.376944,0.690417,0.479157,0.261519,0.989371,0.026807,0.734895}, // user id=41 (ecgid/person_42_rec_1_ecg.csv)
    {0.072222,0.252667,0.333333,0.380889,0.774000,0.354927,0.084983,1.000000,0.002151,0.587992}, // user id=42 (ecgid/person_43_rec_1_ecg.csv)
    {0.107037,0.257778,0.333333,0.390185,0.732222,0.366828,0.156880,0.997585,0.046430,0.517064}, // user id=43 (ecgid/person_44_rec_1_ecg.csv)
    {0.132500,0.239583,0.333333,0.386250,0.730417,0.418626,0.072691,1.000000,0.001799,0.481614}, // user id=44 (ecgid/person_45_rec_1_ecg.csv)
    {0.138788,0.268939,0.333333,0.359697,0.701818,0.300690,0.143796,1.000000,0.000000,0.468543}, // user id=45 (ecgid/person_46_rec_1_ecg.csv)
    {0.159130,0.247101,0.333333,0.360580,0.667681,0.306328,0.177979,1.000000,0.000000,0.451001}, // user id=46 (ecgid/person_47_rec_1_ecg.csv)
    {0.114694,0.180204,0.333333,0.522449,0.777211,0.517179,0.240061,0.965422,0.009289,0.675917}, // user id=47 (ecgid/person_48_rec_1_ecg.csv)
    {0.081037,0.190370,0.333333,0.496963,0.781630,0.752979,0.194462,0.882192,0.003395,0.887803}, // user id=48 (ecgid/person_49_rec_1_ecg.csv)
    {0.144561,0.257719,0.333333,0.406491,0.666316,0.347379,0.010121,1.000000,0.030200,0.539781}, // user id=49 (ecgid/person_50_rec_1_ecg.csv)
    {0.101961,0.257059,0.333333,0.378235,0.781569,0.548395,0.226968,0.962879,0.000000,0.908654}, // user id=50 (ecgid/person_51_rec_1_ecg.csv)
    {0.148000,0.279167,0.333333,0.362500,0.738833,0.384442,0.113496,1.000000,0.000000,0.523396}, // user id=51 (ecgid/person_52_rec_1_ecg.csv)
    {0.131746,0.282381,0.333333,0.359048,0.731429,0.373971,0.010582,1.000000,0.054514,0.359725}, // user id=52 (ecgid/person_53_rec_1_ecg.csv)
    {0.120476,0.264762,0.333333,0.372857,0.745952,0.265676,0.121189,1.000000,0.000000,0.553113}, // user id=53 (ecgid/person_54_rec_1_ecg.csv)
    {0.134444,0.264815,0.333333,0.360556,0.787407,0.273869,0.088453,1.000000,0.000000,0.446707}, // user id=54 (ecgid/person_55_rec_1_ecg.csv)
    {0.082963,0.268148,0.333333,0.453889,0.759815,0.246077,0.013733,1.000000,0.037363,0.399243}, // user id=55 (ecgid/person_56_rec_1_ecg.csv)
    {0.121449,0.250725,0.333333,0.397391,0.680145,0.202441,0.000000,1.000000,0.048686,0.362474}, // user id=56 (ecgid/person_57_rec_1_ecg.csv)
    {0.095417,0.265833,0.333333,0.384583,0.756250,0.200351,0.032333,1.000000,0.002281,0.435691}, // user id=57 (ecgid/person_58_rec_1_ecg.csv)
    {0.153485,0.286667,0.333333,0.377879,0.733030,0.578530,0.314558,1.000000,0.000000,0.747655}, // user id=58 (ecgid/person_59_rec_1_ecg.csv)
    {0.095694,0.255139,0.333333,0.425139,0.687917,0.214292,0.007446,0.984887,0.045981,0.619005}, // user id=59 (ecgid/person_60_rec_1_ecg.csv)
    {0.104800,0.285467,0.333333,0.369067,0.707200,0.390821,0.112631,1.000000,0.000000,0.690165}, // user id=60 (ecgid/person_61_rec_1_ecg.csv)
    {0.141667,0.222407,0.333333,0.396481,0.743704,0.423090,0.092248,0.941421,0.006714,0.954501}, // user id=61 (ecgid/person_62_rec_1_ecg.csv)
    {0.117451,0.252941,0.333333,0.361177,0.718431,0.297293,0.077985,1.000000,0.000000,0.706839}, // user id=62 (ecgid/person_63_rec_1_ecg.csv)
    {0.120702,0.272632,0.333333,0.364737,0.745088,0.239333,0.060230,1.000000,0.000777,0.385973}, // user id=63 (ecgid/person_64_rec_1_ecg.csv)
    {0.130556,0.287639,0.333333,0.370833,0.695139,0.392796,0.216649,1.000000,0.000000,0.702700}, // user id=64 (ecgid/person_65_rec_1_ecg.csv)
    {0.117333,0.246111,0.333333,0.397889,0.702778,0.597046,0.363898,0.975067,0.025308,0.830660}, // user id=65 (ecgid/person_66_rec_1_ecg.csv)
    {0.130000,0.279630,0.333333,0.370000,0.750741,0.343966,0.183980,1.000000,0.000000,0.445425}, // user id=66 (ecgid/person_67_rec_1_ecg.csv)
    {0.161154,0.242821,0.333333,0.383718,0.710256,0.577957,0.156384,0.853165,0.000000,0.958507}, // user id=67 (ecgid/person_68_rec_1_ecg.csv)
    {0.121282,0.242564,0.333333,0.388077,0.673974,0.226179,0.014283,1.000000,0.018323,0.406824}, // user id=68 (ecgid/person_69_rec_1_ecg.csv)
    {0.126061,0.293333,0.333333,0.380303,0.752727,0.434234,0.057875,0.991841,0.010695,0.825712}, // user id=69 (ecgid/person_70_rec_1_ecg.csv)
    {0.108833,0.216500,0.333333,0.381000,0.726500,0.529321,0.116078,0.896493,0.000926,0.878521}, // user id=70 (ecgid/person_71_rec_1_ecg.csv)
    {0.117500,0.240500,0.333333,0.370667,0.725667,0.335912,0.197998,1.000000,0.000000,0.637666}, // user id=71 (ecgid/person_72_rec_1_ecg.csv)
    {0.105447,0.169431,0.333333,0.528374,0.788374,0.637837,0.127642,0.872134,0.003151,0.839206}, // user id=72 (ecgid/person_73_rec_1_ecg.csv)
    {0.097917,0.202500,0.333333,0.414583,0.810000,0.379878,0.048688,1.000000,0.009243,0.645313}, // user id=73 (ecgid/person_74_rec_1_ecg.csv)
    {0.154062,0.220000,0.333333,0.367188,0.652396,0.500206,0.403869,1.000000,0.000000,0.586002}, // user id=74 (ecgid/person_75_rec_1_ecg.csv)
    {0.130351,0.238597,0.333333,0.374386,0.709824,0.570770,0.319919,1.000000,0.000000,0.811718}, // user id=75 (ecgid/person_76_rec_1_ecg.csv)
    {0.111667,0.278889,0.333333,0.412778,0.650000,0.213929,0.005955,0.851564,0.025740,0.590823}, // user id=76 (ecgid/person_77_rec_1_ecg.csv)
    {0.106296,0.290556,0.333333,0.473518,0.742963,0.238166,0.000000,1.000000,0.144586,0.310825}, // user id=77 (ecgid/person_78_rec_1_ecg.csv)
    {0.055463,0.155093,0.333333,0.515000,0.819815,0.343923,0.084410,0.924228,0.003949,0.706575}, // user id=78 (ecgid/person_79_rec_1_ecg.csv)
    {0.049306,0.210833,0.333333,0.368333,0.696389,0.413882,0.123322,1.000000,0.000000,0.608080}, // user id=79 (ecgid/person_80_rec_1_ecg.csv)
    {0.146863,0.247059,0.333333,0.369216,0.809608,0.598618,0.181790,0.866955,0.024801,0.940989}, // user id=80 (ecgid/person_81_rec_1_ecg.csv)
    {0.097917,0.223750,0.333333,0.380000,0.674167,0.366680,0.040337,1.000000,0.015240,0.660683}, // user id=81 (ecgid/person_82_rec_1_ecg.csv)
    {0.110667,0.226333,0.333333,0.413833,0.715167,0.319241,0.107831,0.936457,0.000000,0.491870}, // user id=82 (ecgid/person_83_rec_1_ecg.csv)
    {0.060682,0.179848,0.333333,0.480303,0.762046,0.778796,0.361098,0.945505,0.070085,0.878837}, // user id=83 (ecgid/person_84_rec_1_ecg.csv)
    {0.117451,0.183725,0.333333,0.403333,0.749020,0.170094,0.005361,0.784687,0.184146,0.727668}, // user id=84 (ecgid/person_85_rec_1_ecg.csv)
    {9.000000,9.000000,9.000000,9.000000,9.000000,9.000000,9.000000,9.000000,9.000000,9.000000}, // user id=85 (ecgid/person_86_rec_1_ecg.csv)
    {0.082407,0.211852,0.333333,0.432963,0.740926,0.309057,0.003045,1.000000,0.057845,0.661626}, // user id=86 (ecgid/person_87_rec_1_ecg.csv)
    {0.081404,0.168070,0.333333,0.411930,0.785263,0.570020,0.027666,0.830533,0.079844,0.860993}, // user id=87 (ecgid/person_88_rec_1_ecg.csv)
    {0.076019,0.177500,0.333333,0.548611,0.789444,0.869244,0.033799,0.984070,0.001967,0.930204}, // user id=88 (ecgid/person_89_rec_1_ecg.csv)
    {0.098437,0.179792,0.333333,0.491146,0.798542,0.649532,0.100751,0.791858,0.021504,0.918400}  // user id=89 (ecgid/person_90_rec_1_ecg.csv)
};

// - - - - - - - - - - - - - - - - - -

// loads the templates above into a global variable that is used in step5_matching
void step0_loadTemplates(){
    LIST_TEMPLATES_LENGTH = 0;
    printf("step0_loadTemplates starting...\n");
    for(int userId = 0; userId<NUMBER_OF_USERS; userId++){
      LIST_TEMPLATES[userId] = template_user[userId];
      LIST_TEMPLATES_LENGTH++;
    }
    printf("step0_loadTemplates ended. Loaded %d templates.\n\n", LIST_TEMPLATES_LENGTH);
}


typedef struct {
  float * template;
  int userId;
} UserObj;



// this is the core function that runs reads the ECG signal and runs the 4 blocks (filtering, rpeak/segmentation, features extraction, matching)
UserObj processECGFromFile(char filename[], bool isAuthentication){

  bool result = step1_readSignal(filename);

  if (result == false){
    printf("Error trying to read file (%s)!!! Please check if file exists!\n", filename);
    UserObj obj;
    float * invalidTemplate = (float*) calloc (NUMBER_OF_FEATURES, sizeof(float));
    for(int i=0; i<NUMBER_OF_FEATURES; i++) invalidTemplate[i] = 9.000000;
    obj.template = invalidTemplate;
    obj.userId = -1;
    return obj;
  }

  long int * signalFiltered = step2_filtering(); // Block 1

  step3_segmenter(signalFiltered);  // Block 2

  float * userTemplate = step4_features_extraction();  // Block 3

  UserObj obj;
  obj.template = userTemplate;
  obj.userId = -1;

  if (isAuthentication){
    if (LIST_TEMPLATES_LENGTH == 0) {
      printf("Error - LIST_TEMPLATES_LENGTH = 0. Have you called step0_loadTemplates() ? \n");
      return obj;
    }
    obj.userId = step5_matching(userTemplate); // Block 4
  }

  // reseting/clearing variables
  free(signalFiltered);
  R_PEAKS_LENGTH = 0;
  SAMPLE_COUNT = 0;

  return obj;
}

// returns the template (features vector) generated for that user based on the ECG read from filename
UserObj enrollUser(char filename[]){
  return processECGFromFile(filename, false);
}

// returns the UserObj that matched the ECG read from filename
UserObj authenticateUser(char filename[]){
  return processECGFromFile(filename, true);
}


char * getFilenameFor(int userId, char prefix[1024], char postfix[20], bool padCounter, int startFilenameCounterAt){
    int i = userId + startFilenameCounterAt;
    int len = 1; if (i >= 10) len = 2;
    char intChar[len+1];
    if (padCounter)
      sprintf(intChar,"%02d", i);
    else
      sprintf(intChar,"%d", i);
    fflush(stdout);
    // char filename[1024] = "";
    char * filename = (char *) calloc (strlen(prefix) + strlen(postfix) + len, sizeof(char));
    strcat (filename, prefix);
    strcat (filename, intChar);
    strcat (filename, postfix);
    return filename;
}

// creates templates for all users and prints to the console the code that needs to be copy/pasted inside this file to update the templates database
void enrollAllUsers(char prefix[1024], char postfix[20], bool padCounter, int startFilenameCounterAt){
  IS_VERBOSE = false;
  bool print = !DISABLE_PRINTS_FOR_SIMULATION; // disable the prints in this function
  int numberOfUsers = NUMBER_OF_USERS;

  // -- Clearing templates since it will be updated
  LIST_TEMPLATES_LENGTH = 0;
  free(LIST_TEMPLATES);
  LIST_TEMPLATES = (float**) calloc (NUMBER_OF_USERS, sizeof(float *));
  for (int i=0; i< NUMBER_OF_USERS; i++)
      LIST_TEMPLATES[i] = (float*) calloc (NUMBER_OF_FEATURES, sizeof(float));

  if (print){
    printf("// - - - - - copy code below inside the source code to update the features database - - - - - \n\n");
    printf("// WINDOW_SIZE=%d - BUFFER_SIZE=%d\n", WINDOW_SIZE, BUFFER_SIZE);
    printf("float template_user[%d][%d]={\n", NUMBER_OF_USERS, NUMBER_OF_FEATURES);
  }
  for(int i=0;i<NUMBER_OF_USERS;i++){

    char * filename = getFilenameFor(i, prefix,  postfix, padCounter, startFilenameCounterAt);
    // printf("%s\n", filename);
    UserObj user = enrollUser(filename);
    if (print)
      printf("    {");
    if (print){
      // printf("    {%f,%f,%f,%f,%f,%f,%f,%f,%f,%f}", user.template[0], user.template[1],user.template[2],user.template[3],user.template[4],user.template[5],user.template[6],user.template[7],user.template[8],user.template[9]);
      for (int f=0; f<NUMBER_OF_FEATURES; f++){
        float featureValue = user.template[f];
        if isnan(featureValue) featureValue = 9; // default value in case of a NaN
        printf("%f", featureValue);
        if (f < NUMBER_OF_FEATURES-1) printf(",");
      }
      if (print){
        printf("}");
        if (i<NUMBER_OF_USERS-1) printf(","); else printf(" ");
        printf(" // user id=%d (%s)\n", i, filename);
        fflush(stdout);
      }
    }
    free(filename);

    // updating the list template for this session just in case if we want to run the authenticateAllUsers() in sequence
    for(int k=0;k<NUMBER_OF_FEATURES;k++){
      if (!isnan(user.template[k]))
        LIST_TEMPLATES[i][k] = user.template[k];
      else
        LIST_TEMPLATES[i][k] = 9.0;
    }
    free(user.template);
    LIST_TEMPLATES_LENGTH++;
  }
  if (print){
    printf("};\n");
    printf("\n// - - - - - - - - - end of template database code - - - - - - - - - - \n\n");
  }
}

void enrollAllUsers_Fantasia(){
    enrollAllUsers("ecg_for_template_generation/ecg_user_" , ".txt", false, 0);
}

// creates templates for all users and prints to the console the code that needs to be copy/pasted inside this file to update the templates database
void enrollAllUsers_ECGID(){
  // enrollAllUsers("ecgid/3_ecg_raw/person_" , "_rec_1_ecg.csv", true, 1);
  enrollAllUsers("ecgid/4_ecg_filtered/person_" , "_rec_1_ecg.csv", true, 1);
}

// authenticate all users (from ecg_for_testing folder) and provides a report about the accuracy of the system
float authenticateAllUsers(char prefix[1024], char postfix[20], bool padCounter, int startFilenameCounterAt){
  IS_VERBOSE = false;
  bool print = !DISABLE_PRINTS_FOR_SIMULATION;
  int numberOfUsers = NUMBER_OF_USERS;
  int numCorrect = 0, numFailed = 0;

  for(int i=0;i<numberOfUsers;i++){
    float fv = LIST_TEMPLATES[i][0];
    if (fv == 9) { // user not enrolled
      // printf("USER NOT ENROLEED!!!");
      printf("%d) N/A - User %d not enrolled\n", i, i);
      continue;
    }

    char * filename = getFilenameFor(i, prefix,  postfix, padCounter, startFilenameCounterAt);
    // printf("%s\n", filename);
    UserObj user = authenticateUser(filename);
    // printf("float template_user_%d[10] = {%f,%f,%f,%f,%f,%f,%f,%f,%f,%f};\n", i, user.template[0], user.template[1],user.template[2],user.template[3],user.template[4],user.template[5],user.template[6],user.template[7],user.template[8],user.template[9]);
    if (user.userId == i){
      numCorrect++;
      if (print)
        printf("%d) Correct!\n", i);
    } else if (user.userId == -1) {
      if (print)
        printf("%d) N/A - Failed to generate template for user (%s)\n", i, filename);
    } else {
      numFailed++;
      if (print)
        printf("%d) Wrong - Returned %d instead of %d (%s)\n", i, user.userId, i, filename);
    }
    free(filename);
    free(user.template);
  }
  float acc = 1.0*numCorrect / (numCorrect+numFailed);
  if (print){
    printf("-------------------------\n");
    printf("Authentication results (WINDOW SIZE=%d, BUFFER_SIZE=%d):\n", WINDOW_SIZE,BUFFER_SIZE);
    printf("# Correct=%d\n", numCorrect);
    printf("# Failed=%d\n", numFailed);
    printf("Accuracy=%f\n", acc);
    printf("-------------------------\n");
  }

  return acc;
}
float authenticateAllUsers_Fantasia(){
  return authenticateAllUsers("ecg_for_testing/ecg_user_", "_test.txt", false, 0);
}

// authenticate all users (from ECGID directory, recording #2) and provides a report about the accuracy of the system
float authenticateAllUsers_ECGID(){
  // return authenticateAllUsers("ecgid/3_ecg_raw/person_", "_rec_2_ecg.csv", true, 1);
  return authenticateAllUsers("ecgid/4_ecg_filtered/person_", "_rec_2_ecg.csv", true, 1);
}

void reset(){
  if (SAMPLING_RATE == -1){
    printf("Error - SAMPLING_RATE not set. Please call the initialize_xxxx() function");
    return ;
  }

  free(SIGNAL);
  SIGNAL = (long int*) calloc (SIGNAL_SIZE_IN_SECS * SAMPLING_RATE, sizeof(long int));
  free(R_PEAKS);
  R_PEAKS = (long int*) calloc (SIGNAL_SIZE_IN_SECS * NUMBER_OF_FEATURES, sizeof(long int));

  SIGNAL_LENGTH = -1;
  R_PEAKS_LENGTH = 0;
  SAMPLE_COUNT = 0;

  // for (int i=0; i<NUMBER_OF_USERS; i++)
  //   free(LIST_TEMPLATES[i]);
  free(LIST_TEMPLATES);
  LIST_TEMPLATES = (float**) calloc (NUMBER_OF_USERS, sizeof(float *));
  for (int i=0; i< NUMBER_OF_USERS; i++)
      LIST_TEMPLATES[i] = (float*) calloc (NUMBER_OF_FEATURES, sizeof(float));
}




void initialize_for_Fantasia(){
  SIGNAL_SIZE_IN_SECS = 30;
  NUMBER_OF_USERS = 40;
  SAMPLING_RATE = 250;
  WINDOW_SIZE = 12;
  BUFFER_SIZE = 415;

  reset();
}

void initialize_for_ECGID(){
  SIGNAL_SIZE_IN_SECS = 30;
  NUMBER_OF_USERS = 90;
  SAMPLING_RATE = 500;
  WINDOW_SIZE = 13;
  BUFFER_SIZE = 520;

  reset();
}

void initialize_for_ECGID_with_custom(int windowSize, int bufferSize){
  SIGNAL_SIZE_IN_SECS = 30;
  NUMBER_OF_USERS = 90;
  SAMPLING_RATE = 500;
  WINDOW_SIZE = windowSize;
  BUFFER_SIZE = bufferSize;

  reset();
}


int main() {

  printf("Starting ECG Authenticator... \n\n");

  step0_loadTemplates();

  // // example of enrolling an user
  // enrollUser("ecg_for_template_generation/ecg_user_0.txt");
  // enrollUser("person_90_rec_1_ecg.csv");


  // // example of authenticating an user
  // authenticateUser("ecg_for_testing/ecg_user_0_test.txt"); // this test authenticate correctly, returning user id 0
  // authenticateUser("ecg_for_testing/ecg_user_6_test.txt");  // this test fails, since it retuns user id 26 (instead of 6)



  // - - - - - - - - - - - - - - - - - - - - - -
  // // other helper functions


  // initialize_for_Fantasia();
  // enrollAllUsers_Fantasia();
  // float acc = authenticateAllUsers_Fantasia();


  initialize_for_ECGID();
  enrollAllUsers_ECGID();
  float acc = authenticateAllUsers_ECGID();





  printf("\nDone! \n");
  return 0;
}

