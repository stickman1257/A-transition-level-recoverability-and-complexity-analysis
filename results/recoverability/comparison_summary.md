# Recoverability Comparison Report

Models: conditional_unet, cnn_1d, lstm, gru

## notch->bandpass

Best model by CCC: `conditional_unet`

| model | ccc | pearson_r | nrmse | psd_distance | rms_relative_error | median_frequency_relative_error |
|---|---:|---:|---:|---:|---:|---:|
| conditional_unet | 0.986 | 0.986 | 0.016 | 0.102 | 0.022 | 0.042 |
| lstm | 0.985 | 0.986 | 0.016 | 0.103 | 0.023 | 0.038 |
| gru | 0.985 | 0.986 | 0.016 | 0.102 | 0.022 | 0.034 |
| cnn_1d | 0.985 | 0.985 | 0.016 | 0.104 | 0.022 | 0.042 |

## rectified->notch

Best model by CCC: `conditional_unet`

| model | ccc | pearson_r | nrmse | psd_distance | rms_relative_error | median_frequency_relative_error |
|---|---:|---:|---:|---:|---:|---:|
| conditional_unet | 0.008 | 0.010 | 0.173 | 2.010 | 0.602 | 0.339 |
| cnn_1d | 0.005 | 0.007 | 0.168 | 2.641 | 0.677 | 0.544 |
| gru | 0.003 | 0.007 | 0.121 | 1.327 | 0.725 | 0.770 |
| lstm | 0.003 | 0.004 | 0.126 | 1.267 | 0.686 | 0.741 |

## lp_10hz->rectified

Best model by CCC: `gru`

| model | ccc | pearson_r | nrmse | psd_distance | rms_relative_error | median_frequency_relative_error |
|---|---:|---:|---:|---:|---:|---:|
| gru | 0.575 | 0.631 | 0.117 | 0.763 | 0.203 | 0.844 |
| lstm | 0.569 | 0.625 | 0.118 | 0.780 | 0.207 | 0.844 |
| cnn_1d | 0.560 | 0.605 | 0.125 | 0.935 | 0.192 | 0.843 |
| conditional_unet | 0.558 | 0.602 | 0.134 | 2.006 | 0.227 | 0.841 |
