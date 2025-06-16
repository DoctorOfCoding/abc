## Instructions for Github copilot to perform task:

- Tum ek expert data scientist ki tarah kaam karo gy, pure professional tareeqe se.

- Kaggle Competition - "CMI - Detect Behavior with Sensor Data":

- Men is competition ko jeetna chahta hon, aur tum mere liye pura project A to Z banao gy.

**Python & Libraries:**

- Sirf Python (NumPy, Pandas, Scikit-learn, XGBoost, TensorFlow/PyTorch - jo bhi best lage) use karna hai.

- Men khud se koi logic, feature engineering, ya kuch aur nahi karnon ga – tum sab handle karo gy.

- Copy-Paste Ready Solution:

- Men bas tumhary diye hue code ko Kaggle notebook mein copy-paste karke run karon ga.

- High Accuracy Model:

- Model ki accuracy bohat high honi chahiye (top-tier predictions).

- Competition jeetne ke liye best possible approach use karna hai.

**Step-by-Step Cells:**

- Tum har step ko alag-alag cells mein do gy (numbered).

- Har step ke baad tum batao gy ke "Agy ham ne  kya karna hai" aur "Kyun karna hai" phir meri ijazat se agla reply karo gy.

**Roman Urdu Only:**

- Mujh se sirf Roman Urdu mein baat karo gy, Normal Urdu use nahi karo gy.

- Code ke comments English mein likhna (taki Kaggle pe sab clear rahe).

- Short markdown file b har cell k number wise sath sath tum bana k do gy.

- Koi Dataset tum ko chahiye ho to wo tum mujh se maang sakty ho ya koi b cheez jo required ho tum mujh se maang sakty ho Project k liye.

- men ye kaggle notebook py kaam karon ga.

- Har cell ke baad mere next command ka wait karo gy (jese interview ho raha ho one on one).

---


## Project Details

### CMI - Detect Behavior with Sensor Data
Predicting Body Focused Repetitive Behaviors from a Wrist-Worn Device

**Overview**
Can you use movement, temperature, and proximity sensor data to differentiate between body-focused repetitive behaviors (BFRBs), like hair pulling, from non-BFRB everyday gestures, like adjusting glasses? The goal of this competition is to develop a predictive model that distinguishes BFRB-like and non-BFRB-like activity using data from a variety of sensors collected via a wrist-worn device. Successfully disentangling these behaviors will improve the design and accuracy of wearable BFRB-detection devices, which are relevant to a wide range of mental illnesses, ultimately strengthening the tools available to support their treatment.

**Description**
Body-focused repetitive behaviors (BFRBs), such as hair pulling, skin picking, and nail biting, are self-directed habits involving repetitive actions that, when frequent or intense, can cause physical harm and psychosocial challenges. These behaviors are commonly seen in anxiety disorders and obsessive-compulsive disorder (OCD), thus representing key indicators of mental health challenges.

[hand](hand.png)

To investigate BFRBs, the Child Mind Institute has developed a wrist-worn device, Helios, designed to detect these behaviors. While many commercially available devices contain Inertial Measurement Units (IMUs) to measure rotation and motion, the Helios watch integrates additional sensors, including 5 thermopiles (for detecting body heat) and 5 time-of-flight sensors (for detecting proximity). See the figure to the right for the placement of these sensors on the Helios device.

We conducted a research study to test the added value of these additional sensors for detecting BFRB-like movements. In the study, participants performed series of repeated gestures while wearing the Helios device:

1. They began a transition from “rest” position and moved their hand to the appropriate location (Transition);

2. They followed this with a short pause wherein they did nothing (Pause); and

3. Finally they performed a gesture from either the BFRB-like or non-BFRB-like category of movements (Gesture; see Table below).

Each participant performed 18 unique gestures (8 BFRB-like gestures and 10 non-BFRB-like gestures) in at least 1 of 4 different body-positions (sitting, sitting leaning forward with their non-dominant arm resting on their leg, lying on their back, and lying on their side). These gestures are detailed in the table below, along with a video of the gesture.

BFRB-Like Gesture (Target Gesture) | Video Example
------ | ---------
Above ear - Pull hair | Sitting
Forehead - Pull hairline | Sitting leaning forward
Forehead - Scratch | Sitting
Eyebrow - Pull hair | Sitting
Eyelash - Pull hair | Sitting
Neck - Pinch skin | Sitting
Neck - Scratch | Sitting
Cheek - Pinch skin | Sitting Sitting leaning forward, Lying on back, Lying on side


Non-BFRB-Like Gesture (Non-Target Gesture) | Video Example
------- | ------
Drink from bottle/cup | Sitting
Glasses on/off | Sitting
Pull air toward your face | Sitting
Pinch knee/leg skin | Sitting leaning forward
Scratch knee/leg skin | Sitting leaning forward
Write name on leg | Sitting leaning forward
Text on phone | Sitting
Feel around in tray and pull out an object | Sitting
Write name in air | Sitting
Wave hello | Sitting

This competition challenges you to develop a predictive model capable of distinguishing (1) BFRB-like gestures from non-BFRB-like gestures and (2) the specific type of BFRB-like gesture. Critically, when your model is evaluated, half of the test set will include only data from the IMU, while the other half will include all of the sensors on the Helios device (IMU, thermopiles, and time-of-flight sensors).

Your solutions will have direct real-world impact, as the insights gained will inform design decisions about sensor selection — specifically whether the added expense and complexity of thermopile and time-of-flight sensors is justified by significant improvements in BFRB detection accuracy compared to an IMU alone. By helping us determine the added value of these thermopiles and time-of-flight sensors, your work will guide the development of better tools for detection and treatment of BFRBs.

Relevant articles:
Garey, J. (2025). What Is Excoriation, or Skin-Picking? Child Mind Institute. https://childmind.org/article/excoriation-or-skin-picking/
Martinelli, K. (2025). What is Trichotillomania? Child Mind Institute. https://childmind.org/article/what-is-trichotillomania/

**Evaluation**
The evaluation metric for this contest is a version of macro F1 that equally weights two components:

1. Binary F1 on whether the `gesture` is one of the target or `non-target` types.
2. Macro F1 on `gesture`, where all non-target sequences are collapsed into a single non_target class
The final score is the average of the binary F1 and the macro F1 scores.

If your submission includes a `gesture` value not found in the train set your submission will trigger an error.


**Submission File**
You must submit to this competition using the provided evaluation API, which ensures that models perform inference on a single sequence at a time. For each `sequence_id` in the test set, you must predict the corresponding `gesture`.


**Timeline**
* May 29, 2025 - Start Date.

* August 26, 2025 - Entry Deadline. You must accept the competition rules before this date in order to compete.

* August 26, 2025 - Team Merger Deadline. This is the last day participants may join or merge teams.

* September 2, 2025 - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

**Prizes**
* 1st Place - $ 15,000
* 2nd Place - $ 10,000
* 3rd Place - $ 8,000
* 4th Place - $ 7,000
* 5th Place - $ 5,000
* 6th Place - $ 5,000


**Acknowledgements**
The data used for this competition was provided in collaboration with the Healthy Brain Network, a landmark mental health study based in New York City that will help children around the world. In the Healthy Brain Network, families, community leaders, and supporters are partnering with the Child Mind Institute to unlock the secrets of the developing brain. Additional study participants were recruited from Child Mind Institute’s staff and community, and we are grateful for their collaboration. In addition to the generous support provided by the Kaggle team, financial support has been provided by the California Department of Health Care Services (DHCS) as part of the Children and Youth Behavioral Health Initiative (CYBHI).

* **HCS**
CALIFORNIA DEPARTMENT OF
HEALTH CARE SERVICES

* Child Mind Institute Healthy Brain Network



**Code Requirements**

Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

* CPU Notebook <= 9 hours run-time
* GPU Notebook <= 9 hours run-time
* Internet access disabled
* Freely & publicly available external data is allowed, including pre-trained models

Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.


**About the Child Mind Institute and the Healthy Brain Network**

[The Child Mind Institute (CMI)](https://childmind.org/) is the leading independent nonprofit in children’s mental health providing gold-standard, evidence-based care, delivering educational resources to millions of families each year, training educators in underserved communities, and developing open science initiatives and tomorrow’s breakthrough treatments.
[The Healthy Brain Network (HBN)](https://healthybrainnetwork.org/) is a community-based research initiative of the Child Mind Institute. We provide no-cost, study-related mental health and learning evaluations to children ages 5–21 and connect families with community resources. We are collecting the information needed to find brain and body characteristics that are associated with mental health and learning disorders. The Healthy Brain Network stores and openly shares de-identified data about psychiatric, behavioral, cognitive, and lifestyle (e.g., fitness, diet) phenotypes, as well as multimodal brain imaging (MRI), electroencephalography (EEG), digital voice and video recordings, genetics, and actigraphy.



**Citation**
Laura Newman, David LoBue, Arianna Zuanazzi, Florian Rupprecht, Luke Mears, Roxanne McAdams, Erin Brown, Yanyi Wang, Camilla Strauss, Arno Klein, Lauren Hendrix, Maki Koyama, Josh To, Curt White, Yuki Kotani, Michelle Freund, Michael Milham, Gregory Kiar, Martyna Plomecka, Sohier Dane, and Maggie Demkin. CMI - Detect Behavior with Sensor Data. https://kaggle.com/competitions/cmi-detect-behavior-with-sensor-data, 2025. Kaggle.



**Dataset Description**
In this competition you will use sensor data to classify body-focused repetitive behaviors (BFRBs) and other gestures.

This dataset contains sensor recordings taken while participants performed 8 BFRB-like gestures and 10 non-BFRB-like gestures while wearing the Helios device on the wrist of their dominant arm. The Helios device contains three sensor types:

1. 1x Inertial Measurement Unit (IMU; BNO080/BNO085): An integrated sensor that combines accelerometer, gyroscope, and magnetometer measurements with onboard processing to provide orientation and motion data.
2. 5x Thermopile Sensor (MLX90632): A non-contact temperature sensor that measures infrared radiation.
3. 5x Time-of-Flight Sensor (VL53L7CX): A sensor that measures distance by detecting how long it takes for emitted infrared light to bounce back from objects.

You must submit to this competition using the provided Python evaluation API, which serves test set data one sequence at a time. To use the API, follow the example in this notebook.

Expect approximately 3,500 sequences in the hidden test set.
Half of the hidden-test sequences are recorded with IMU only; the thermopile (thm_) and time-of-flight (tof__v*) columns are still present but contain null values for those sequences.
This will allow us to determine whether adding the time-of-flight and thermopile sensors improves our ability to detect BFRBs. Note also that there is known sensor communication failure in this dataset, resulting in missing data from some sensors in some sequences.

**Files**
*[train/test].csv*

* `row_id`
* `sequence_id` - An ID for the batch of sensor data. Each sequence includes one Transition, one Pause, and one Gesture.
* `sequence_type` - If the gesture is a target or non-target type. Train only.
* `sequence_counter` - A counter of the row within each sequence.
* `subject` - A unique ID for the subject who provided the data.
* `gesture` - The target column. Description of sequence Gesture. Train only.
* `orientation` - Description of the subject's orientation during the sequence. Train only.
* `behavior` - A description of the subject's behavior during the current phase of the sequence.
* `acc_[x/y/z]` - Measure linear acceleration along three axes in meters per second squared from the IMU sensor.
* `rot_[w/x/y/z]` - Orientation data which combines information from the IMU's gyroscope, accelerometer, and magnetometer to describe the device's orientation in 3D space.
* `thm_[1-5]` - There are five thermopile sensors on the watch which record temperature in degrees Celsius. Note that the index/number for each corresponds to the index in the photo on the Overview tab.
* `tof_[1-5]_v[0-63]` - There are five time-of-flight sensors on the watch that measure distance. In the dataset, the 0th pixel for the first time-of-flight sensor can be found with column name `tof_1_v0`, whereas the final pixel in the grid can be found under column `tof_1_v63`. This data is collected row-wise, where the first pixel could be considered in the top-left of the grid, with the second to its right, ultimately wrapping so the final value is in the bottom right (see image above). The particular time-of-flight sensor is denoted by the number at the start of the column name (e.g., 1_v0 is the first pixel for the first time-of-flight sensor while 5_v0 is the first pixel for the fifth time-of-flight sensor). If there is no sensor response (e.g., if there is no nearby object causing a signal reflection), a -1 is present in this field. Units are uncalibrated sensor values in the range 0-254. Each sensor contains 64 pixels arranged in an 8x8 grid, visualized in the figure below.

[image](image.png)

**[train/test]_demographics.csv**

These tabular files contain demographic and physical characteristics of the participants.
subject
* `adult_child`: Indicates whether the participant is a child (`0`) or an adult (`1`). Adults are defined as individuals * aged 18 years or older.
* `age`: Participant's age in years at time of data collection.
* `sex`: Participants sex assigned at birth, `0`= female, `1` = male.
* `handedness`: Dominant hand used by the participant, `0` = left-handed, `1` = right-handed.
* `height_cm`: Height of the participant in centimeters.
* `shoulder_to_wrist_cm`: Distance from shoulder to wrist in centimeters.
* `elbow_to_wrist_cm`: Distance from elbow to wrist in centimeters.

**sample_submission.csv**

* `sequence_id`
* `gesture`

**kaggle_evaluation/** The files that implement the evaluation API. You can run the API locally on a Unix machine for testing purposes.


## Dataset k Acuatll Paths

Datasets k actuall paths.
* [cmi-detect-behavior-with-sensor-data](/kaggle/input/cmi-detect-behavior-with-sensor-data):
    * file [test.csv](/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv)
    * file [test_demographics.csv](/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv)
    * file [train.csv](/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv)
    * file [train_demographics.csv](/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv)
    * kaggle_evaluation folder [kaggle_evaluation](/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation):
        * file [__init__.py](/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation/__init__.py)
        * file [cmi_gateway.py](/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation/cmi_gateway.py)
        * file [cmi_inference_server.py](/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation/cmi_inference_server.py)
        * core subfoler [core](/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation/core):
            * file [__init__.py](/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation/core/__init__.py)
            * file [base_gateway.py](/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation/core/base_gateway.py)
            * file [kaggle_evaluation.proto](/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation/core/kaggle_evaluation.proto)
            * file [relay.py](/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation/core/relay.py)
            * file [templates.py](/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation/core/templates.py) 
            * generated subfolder [generated](/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation/core/generated)
                * file [__init__.py](/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation/core/generated/__init__.py)
                * file [kaggle_evaluation_pb2.py](/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation/core/generated/kaggle_evaluation_pb2.py)
                * file [kaggle_evaluation_pb2_grpc.py](/kaggle/input/cmi-detect-behavior-with-sensor-data/kaggle_evaluation/core/generated/kaggle_evaluation_pb2_grpc.py)


**test.csv** columns and fews rows
| row_id | sequence_id | sequence_counter | subject | acc_x | acc_y | acc_z | rot_w | rot_x | rot_y |
|-----|------|------|------|--------|--------|------|-------|------|------|
SEQ_000001_000000 | SEQ_000001 | 0 | SUBJ_055840 | 9.0390625 | 5.26171875 | 0.80078125 | 0.3671875 | -0.39739990234375 | -0.6290283203125 |
SEQ_000001_000001 | SEQ_000001 | 1 | SUBJ_055840 | 9.421875 | 3.4609375 | -1.11328125 | 0.3538818359375 | -0.50714111328125 | -0.6527099609375 |
SEQ_000001_000002 | SEQ_000001 | 2 | SUBJ_055840 | 10.16015625 | 2.08203125 | -3.87109375 | 0.38409423828125 | -0.5321044921875 | -0.6396484375 |
SEQ_000001_000003 | SEQ_000001 | 3 | SUBJ_055840 | 9.7734375 | 1.35546875 | -4.37109375 | 0.38775634765625 | -0.531982421875 | -0.634033203125 |


**train.csv** columns and fews rows
| row_id | sequence_type | sequence_id | sequence_counter | subject | orientation | behavior | phase | gesture | acc_x | 
|-----|------|------|------|--------|--------|------|-------|------|------|
SEQ_000007_000000 | Target | SEQ_000007 | 0 | SUBJ_059520 | Seated Lean Non Dom - FACE DOWN | Relaxes and moves hand to target location | Transition | Cheek - pinch skin | 6.68359375 |
SEQ_000007_000001 | Target | SEQ_000007 | 1 | SUBJ_059520 | Seated Lean Non Dom - FACE DOWN | Relaxes and moves hand to target location | Transition | Cheek - pinch skin | 6.94921875 |
SEQ_000007_000002 | Target | SEQ_000007 | 2 | SUBJ_059520 | Seated Lean Non Dom - FACE DOWN | Relaxes and moves hand to target location | Transition | Cheek - pinch skin | 5.72265625 |
SEQ_000007_000003 | Target | SEQ_000007 | 3 | SUBJ_059520 | Seated Lean Non Dom - FACE DOWN | Relaxes and moves hand to target location | Transition | Cheek - pinch skin | 6.6015625 |


**test_demographics.csv** full content
subject | adult_child | age | sex | handedness | height_cm | shoulder_to_wrist_cm | elbow_to_wrist_cm
-------|------|--------|------|-------|--------|--------|---------
SUBJ_016452 | 1 | 25 | 1 | 1 | 165.0 | 52 | 23.0 |
SUBJ_055840 | 0 | 13 | 0 | 1 | 177.0 | 52 | 27.0 |


**train_demographics.csv** total column and few intial rows
| subject | adult_child | age | sex | handedness | height_cm | shoulder_to_wrist_cm | elbow_to_wrist_cm |
|------|--------|--------|-------|---------|--------|--------|---------|
| SUBJ_000206 | 1 | 41 | 1 | 1 | 172.0 | 50 | 25.0 |
| SUBJ_001430 | 0 | 11 | 0 | 1 | 167.0 | 51 | 27.0 |
| SUBJ_002923 | 1 | 28 | 1 | 0 | 164.0 | 54 | 26.0 |


`kaggle_evaluation` k ander ye **cmi_gateway.py** file ka full code
```python
"""Gateway notebook for https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data"""

import os
import random
import time

import polars as pl

import kaggle_evaluation.core.templates

from kaggle_evaluation.core.base_gateway import GatewayRuntimeError, GatewayRuntimeErrorType


class CMIGateway(kaggle_evaluation.core.templates.Gateway):
    def __init__(self, data_paths: tuple[str] | None = None):
        super().__init__(data_paths=data_paths, target_column_name='gesture')
        self.set_response_timeout_seconds(30 * 60)
        self.target_gestures = [
            'Above ear - pull hair',
            'Cheek - pinch skin',
            'Eyebrow - pull hair',
            'Eyelash - pull hair',
            'Forehead - pull hairline',
            'Forehead - scratch',
            'Neck - pinch skin',
            'Neck - scratch',
        ]
        self.non_target_gestures = [
            'Write name on leg',
            'Wave hello',
            'Glasses on/off',
            'Text on phone',
            'Write name in air',
            'Feel around in tray and pull out an object',
            'Scratch knee/leg skin',
            'Pull air toward your face',
            'Drink from bottle/cup',
            'Pinch knee/leg skin'
        ]
        self.all_gestures = self.target_gestures + self.non_target_gestures

    def unpack_data_paths(self) -> None:
        if not self.data_paths:
            self.test_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv'
            self.demographics_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'
        else:
            self.test_path = self.data_paths[0]
            self.demographics_path = self.data_paths[1]

    def generate_data_batches(self):
        test = pl.read_csv(self.test_path)
        demos = pl.read_csv(self.demographics_path)

        # Different order for every run.
        # Using time as the seed is the default behavior but it's so important that we'll be explicit.
        random.seed(time.time())

        sequence_ids = list(test['sequence_id'].unique())
        for seq_id in random.sample(sequence_ids, len(sequence_ids)):
            sequence = test.filter(pl.col('sequence_id') == seq_id)
            sequence_demos = demos.filter(pl.col('subject') == sequence['subject'][0])
            yield (sequence, sequence_demos), pl.DataFrame(data={'sequence_id':[seq_id]})

    def validate_prediction_batch(self, prediction: str, row_ids: pl.Series) -> None:
        if not isinstance(prediction, str):
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.INVALID_SUBMISSION, f'Invalid prediction type, expected str but got {type(prediction)}'
            )
        if prediction not in self.all_gestures:
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.INVALID_SUBMISSION, f'All gestures must match those found in the train data. Got {prediction}'
            )
        super().validate_prediction_batch(prediction, row_ids)


if __name__ == '__main__':
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        gateway = CMIGateway()
        # Relies on valid default data paths
        gateway.run()
    else:
        print('Skipping run for now')
```

`kaggle_evaluation` k ander ye **cmi_inference_server.py** file ka full code
```python
import os
import sys

import kaggle_evaluation.core.templates

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import cmi_gateway


class CMIInferenceServer(kaggle_evaluation.core.templates.InferenceServer):
    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None):
        return cmi_gateway.CMIGateway(data_paths)
```

kaggle_evaluation k ander subfolder core k ander ye **__init__.py** file ka full code
```python
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

`kaggle_evaluation` k ander subfolder `core` k ander ye **base_gateway.py** file ka full code
```python
"""Lower level implementation details of the gateway.
Hosts should not need to review this file before writing their competition specific gateway.
"""

import enum
import json
import os
import pathlib
import re
import subprocess

from socket import gaierror
from typing import Any, List, Optional, Tuple, Union

import grpc
import numpy as np
import pandas as pd
import polars as pl

import kaggle_evaluation.core.relay


# Files in this directory are visible to the competitor container.
_FILE_SHARE_DIR = '/kaggle/shared/'
IS_RERUN = os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None


class GatewayRuntimeErrorType(enum.Enum):
    """Allow-listed error types that Gateways can raise, which map to canned error messages to show users.
    Please try capture all errors with one of these types.
    Unhandled errors are treated as caused by Kaggle and do not count towards daily submission limits.
    """

    UNSPECIFIED = 0
    SERVER_NEVER_STARTED = 1
    SERVER_CONNECTION_FAILED = 2
    SERVER_RAISED_EXCEPTION = 3
    SERVER_MISSING_ENDPOINT = 4
    # Default error type if an exception was raised that was not explicitly handled by the Gateway
    GATEWAY_RAISED_EXCEPTION = 5
    INVALID_SUBMISSION = 6
    GRPC_DEADLINE_EXCEEDED = 7


class GatewayRuntimeError(Exception):
    """Gateways can raise this error to capture a user-visible error enum from above and host-visible error details."""

    def __init__(self, error_type: GatewayRuntimeErrorType, error_details: Optional[str] = None):
        self.error_type = error_type
        self.error_details = error_details


class BaseGateway:
    def __init__(self, data_paths: Tuple[str] = None, file_share_dir: Optional[str] = _FILE_SHARE_DIR, target_column_name: Optional[str] = None):
        self.client = kaggle_evaluation.core.relay.Client('inference_server' if IS_RERUN else 'localhost')
        self.server = None  # The gateway can have a server but it isn't typically necessary.
        # Off Kaggle, we can accept a user input file_share_dir. On Kaggle, we need to use the special directory
        # that is visible to the user.
        if file_share_dir or not os.path.exists('/kaggle'):
            self.file_share_dir = file_share_dir
        else:
            self.file_share_dir = _FILE_SHARE_DIR

        self._shared_a_file = False
        self.data_paths = data_paths
        self.target_column_name = (
            target_column_name  # Only used if the predictions are made as a primitive type (int, bool, etc) rather than a dataframe.
        )

    def validate_prediction_batch(self, prediction_batch: Any, row_ids: Union[pl.DataFrame, pl.Series, pd.DataFrame, pd.Series]) -> None:
        """If competitors can submit fewer rows than expected they can save all predictions for the last batch and
        bypass the benefits of the Kaggle evaluation service. This attack was seen in a real competition with the older time series API:
        https://www.kaggle.com/competitions/riiid-test-answer-prediction/discussion/196066
        It's critically important that this check be run every time predict() is called.

        If your predictions may take a variable number of rows and you need to write a custom version of this check,
        you still must specify a minimum row count greater than zero per prediction batch.
        """
        if prediction_batch is None:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, 'No prediction received')
        num_received_rows = None
        # Special handling for numpy ints only as numpy floats are python floats, but numpy ints aren't python ints
        for primitive_type in [int, float, str, bool, np.int_]:
            if isinstance(prediction_batch, primitive_type):
                # Types that only support one predictions per batch don't need to be validated.
                # Basic types are valid for prediction, but either don't have a length (int) or the length isn't relevant for
                # purposes of this check (str).
                num_received_rows = 1
        if num_received_rows is None:
            if type(prediction_batch) not in [pl.DataFrame, pl.Series, pd.DataFrame, pd.Series]:
                raise GatewayRuntimeError(
                    GatewayRuntimeErrorType.INVALID_SUBMISSION, f'Invalid prediction data type, received: {type(prediction_batch)}'
                )
            num_received_rows = len(prediction_batch)
        if type(row_ids) not in [pl.DataFrame, pl.Series, pd.DataFrame, pd.Series]:
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f'Invalid row ID type {type(row_ids)}; expected Polars DataFrame or similar'
            )
        num_expected_rows = len(row_ids)
        if len(row_ids) == 0:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, 'Missing row IDs for batch')
        if num_received_rows != num_expected_rows:
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.INVALID_SUBMISSION, f'Invalid predictions: expected {num_expected_rows} rows but received {num_received_rows}'
            )

    def _standardize_and_validate_paths(self, input_paths: List[Union[str, pathlib.Path]]) -> Tuple[List[str], List[str]]:
        # Accept a list of str or pathlib.Path, but standardize on list of str
        if input_paths and not self.file_share_dir or type(self.file_share_dir) not in (str, pathlib.Path):
            raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f'Invalid `file_share_dir`: {self.file_share_dir}')

        for path in input_paths:
            if os.pardir in str(path):
                raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f'Send files path contains {os.pardir}: {path}')
            if str(path) != str(os.path.normpath(path)):
                # Raise an error rather than sending users unexpectedly altered paths
                raise GatewayRuntimeError(
                    GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f'Send files path {path} must be normalized. See `os.path.normpath`'
                )
            if type(path) not in (pathlib.Path, str):
                raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, 'All paths must be of type str or pathlib.Path')
            if not os.path.exists(path):
                raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f'Input path {path} does not exist')

        input_paths = [os.path.abspath(path) for path in input_paths]
        if len(set(input_paths)) != len(input_paths):
            raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, 'Duplicate input paths found')

        output_dir = str(self.file_share_dir)
        if not output_dir.endswith(os.path.sep):
            # Ensure output dir is valid for later use
            output_dir += os.path.sep

        # Can't use os.path.join for output_dir + path: os.path.join won't prepend to an abspath
        # normpath manages // in particular.
        output_paths = [os.path.normpath(output_dir + path) for path in input_paths]
        return input_paths, output_paths

    def share_files(
        self,
        input_paths: List[Union[str, pathlib.Path]],
    ) -> List[str]:
        """Makes files and/or directories available to the user's inference_server. They will be mirrored under the
        self.file_share_dir directory, using the full absolute path. An input like:
            /kaggle/input/mycomp/test.csv
        Would be written to:
            /kaggle/shared/kaggle/input/mycomp/test.csv

        Args:
            input_paths: List of paths to files and/or directories that should be shared.

        Returns:
            The output paths that were shared.

        Raises:
            GatewayRuntimeError if any invalid paths are passed.
        """
        if self.file_share_dir and not self._shared_a_file:
            if os.path.exists(self.file_share_dir) and (not os.path.isdir(self.file_share_dir) or len(os.listdir(self.file_share_dir)) > 0):
                raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, '`file_share_dir` must be an empty directory.')
            os.makedirs(self.file_share_dir, exist_ok=True)
            self._shared_a_file = True

        if not input_paths:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, 'share_files requires at least one input path')

        input_paths, output_paths = self._standardize_and_validate_paths(input_paths)
        for in_path, out_path in zip(input_paths, output_paths):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # This makes the files available to the InferenceServer as read-only. Only the Gateway can mount files.
            # mount will only work in live kaggle evaluation rerun sessions. Otherwise use a symlink.
            if IS_RERUN:
                if not os.path.isdir(out_path):
                    pathlib.Path(out_path).touch()
                try:
                    subprocess.run(f'mount --bind {in_path} {out_path}', shell=True, check=True)
                except Exception:
                    # `mount`` is expected to be faster but less reliable in our context.
                    # Fall back to cp if possible.
                    if self.file_share_dir != _FILE_SHARE_DIR:
                        raise GatewayRuntimeError(
                            GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION,
                            f'share_files fallback failure: can only use cp if file_share_dir is {_FILE_SHARE_DIR}. Got {self.file_share_dir}',
                        )
                    # cp will fail if the output directory doesn't already exist
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    subprocess.run(f'cp -r {in_path} {out_path}', shell=True, check=True)
            else:
                subprocess.run(f'ln -s {in_path} {out_path}', shell=True, check=True)

        return output_paths

    def write_submission(self, predictions, row_ids: List[Union[pl.Series, pl.DataFrame, pd.Series, pd.DataFrame]]) -> None:
        """Export the predictions to a submission.parquet."""
        if isinstance(predictions, list):
            if isinstance(predictions[0], pd.DataFrame):
                predictions = pd.concat(predictions, ignore_index=True)
            elif isinstance(predictions[0], pl.DataFrame):
                try:
                    predictions = pl.concat(predictions, how='vertical_relaxed')
                except pl.exceptions.SchemaError:
                    raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, 'Inconsistent prediction types')
                except pl.exceptions.ComputeError:
                    raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, 'Inconsistent prediction column counts')
            else:
                if type(row_ids[0]) in [pl.Series, pl.DataFrame]:
                    row_ids = pl.concat(row_ids)
                elif type(row_ids[0]) in [pd.Series, pd.DataFrame]:
                    row_ids = pd.concat(row_ids).reset_index(drop=True)
                else:
                    raise GatewayRuntimeError(
                        GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION,
                        f'Invalid row ID datatype {type(row_ids[0])}. Expected Polars series or dataframe.',
                    )
                if self.target_column_name is None:
                    raise GatewayRuntimeError(
                        GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, '`target_column_name` must be set in order to use scalar value predictions.'
                    )
                predictions = pl.DataFrame(data={row_ids.columns[0]: row_ids, self.target_column_name: predictions})

        if isinstance(predictions, pd.DataFrame):
            predictions.to_parquet('submission.parquet', index=False)
        elif isinstance(predictions, pl.DataFrame):
            pl.DataFrame(predictions).write_parquet('submission.parquet')
        else:
            raise GatewayRuntimeError(
                GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, f"Unsupported predictions type {type(predictions)}; can't write submission file"
            )

    def write_result(self, error: Optional[GatewayRuntimeError] = None) -> None:
        """Export a result.json containing error details if applicable."""
        result = {'Succeeded': error is None}

        if error is not None:
            result['ErrorType'] = error.error_type.value
            result['ErrorName'] = error.error_type.name
            # Max error detail length is 8000
            result['ErrorDetails'] = str(error.error_details[:8000]) if error.error_details else None

        with open('result.json', 'w') as f_open:
            json.dump(result, f_open)

    def handle_server_error(self, exception: Exception, endpoint: str) -> None:
        """Determine how to handle an exception raised when calling the inference server. Typically just format the
        error into a GatewayRuntimeError and raise.
        """
        exception_str = str(exception)
        if isinstance(exception, gaierror) or (isinstance(exception, RuntimeError) and 'Failed to connect to server after waiting' in exception_str):
            raise GatewayRuntimeError(GatewayRuntimeErrorType.SERVER_NEVER_STARTED) from None
        if f'No listener for {endpoint} was registered' in exception_str:
            raise GatewayRuntimeError(GatewayRuntimeErrorType.SERVER_MISSING_ENDPOINT, f'Server did not register a listener for {endpoint}') from None
        if 'Exception calling application' in exception_str:
            # Extract just the exception message raised by the inference server
            message_match = re.search('"Exception calling application: (.*)"', exception_str, re.IGNORECASE)
            message = message_match.group(1) if message_match else exception_str
            raise GatewayRuntimeError(GatewayRuntimeErrorType.SERVER_RAISED_EXCEPTION, message) from None
        if isinstance(exception, grpc._channel._InactiveRpcError):
            raise GatewayRuntimeError(GatewayRuntimeErrorType.SERVER_CONNECTION_FAILED, exception_str) from None
        if isinstance(exception, kaggle_evaluation.core.relay.GRPCDeadlineError):
            raise GatewayRuntimeError(GatewayRuntimeErrorType.GRPC_DEADLINE_EXCEEDED, exception_str) from None

        raise exception
```

`kaggle_evaluation` k ander subfolder `core` k ander ye **kaggle_evaluation.proto** file ka full code
```proto
// Defines the proto service for KaggleEvaluation communication, aiming to provide native
// support for passing a variety of python primitives + common data science
// objects, and nested objects thereof.

syntax = "proto3";

package kaggle_evaluation_client;

service KaggleEvaluationService {
  rpc Send(KaggleEvaluationRequest) returns (KaggleEvaluationResponse) {};
}

message KaggleEvaluationRequest {
  string name = 1;
  // Support generic python method calls using standard args / kwargs format.
  repeated Payload args = 2;
  map<string, Payload> kwargs = 3;
}

message KaggleEvaluationResponse {
  Payload payload = 1;
}

// Core object representing a python value.
message Payload {
  oneof value {
    // Primitives
    string str_value = 1;
    bool bool_value = 2;
    sint64 int_value = 3;
    float float_value = 4;
    // Value is ignored, being set at all means `None`
    bool none_value = 5;

    // Iterables for nested types
    PayloadList list_value = 6;
    PayloadList tuple_value = 7;
    // Only supports dict with keys of type str and values that are serializable
    // to Payload as well.
    PayloadMap dict_value = 8;

    // Allowlisted special types
    // pandas.DataFrame
    bytes pandas_dataframe_value = 9;
    // polars.DataFrame
    bytes polars_dataframe_value = 10;
    // pandas.Series
    bytes pandas_series_value = 11;
    // polars.Series
    bytes polars_series_value = 12;
    // numpy.ndarray
    bytes numpy_array_value = 13;
    // numpy.scalar. Distinct from numpy.ndarray to avoid issues with dimensionless numpy arrays
    bytes numpy_scalar_value = 14;
    // io.BytesIO
    bytes bytes_io_value = 15;
  }
}

message PayloadList {
  repeated Payload payloads = 1;
}

message PayloadMap {
  map<string, Payload> payload_map = 1;
}
```


`kaggle_evaluation` k ander subfolder `core` k ander ye **relay.py** file ka full code
```python
"""
Core implementation of the client module, implementing generic communication
patterns with Python in / Python out supporting many (nested) primitives +
special data science types like DataFrames or np.ndarrays, with gRPC + protobuf
as a backing implementation.
"""

import io
import json
import socket
import time

from concurrent import futures
from typing import Any, Callable, Optional, Tuple

import grpc
import numpy as np
import pandas as pd
import polars as pl
import pyarrow

from grpc._channel import _InactiveRpcError

import kaggle_evaluation.core.generated.kaggle_evaluation_pb2 as kaggle_evaluation_proto
import kaggle_evaluation.core.generated.kaggle_evaluation_pb2_grpc as kaggle_evaluation_grpc


class GRPCDeadlineError(Exception):
    pass


_SERVICE_CONFIG = {
    # Service config proto: https://github.com/grpc/grpc-proto/blob/ec886024c2f7b7f597ba89d5b7d60c3f94627b17/grpc/service_config/service_config.proto#L377
    'methodConfig': [
        {
            'name': [{}],  # Applies to all methods
            # See retry policy docs: https://grpc.io/docs/guides/retry/
            'retryPolicy': {
                'maxAttempts': 5,
                'initialBackoff': '0.1s',
                'maxBackoff': '1s',
                'backoffMultiplier': 1,  # Ensure relatively rapid feedback in the event of a crash
                'retryableStatusCodes': ['UNAVAILABLE'],
            },
        }
    ]
}

# Include potential fallback ports
GRPC_PORTS = [50051] + [i for i in range(60053, 60053 + 10)]

_GRPC_CHANNEL_OPTIONS = [
    # -1 for unlimited message send/receive size
    # https://github.com/grpc/grpc/blob/v1.64.x/include/grpc/impl/channel_arg_names.h#L39
    ('grpc.max_send_message_length', -1),
    ('grpc.max_receive_message_length', -1),
    # https://github.com/grpc/grpc/blob/master/doc/keepalive.md
    ('grpc.keepalive_time_ms', 60_000),  # Time between heartbeat pings
    ('grpc.keepalive_timeout_ms', 5_000),  # Time allowed to respond to pings
    ('grpc.http2.max_pings_without_data', 0),  # Remove another cap on pings
    ('grpc.keepalive_permit_without_calls', 1),  # Allow heartbeat pings at any time
    ('grpc.http2.min_ping_interval_without_data_ms', 1_000),
    ('grpc.service_config', json.dumps(_SERVICE_CONFIG)),
]


DEFAULT_DEADLINE_SECONDS = 60 * 60
_RETRY_SLEEP_SECONDS = 1 / len(GRPC_PORTS)
# Enforce a relatively strict server startup time so users can get feedback quickly if they're not
# configuring KaggleEvaluation correctly. We really don't want notebooks timing out after nine hours
# somebody forgot to start their inference_server. Slow steps like loading models
# can happen during the first inference call if necessary.
STARTUP_LIMIT_SECONDS = 60 * 15

### Utils shared by client and server for data transfer

# pl.Enum is currently unstable, but we should eventually consider supporting it.
# https://docs.pola.rs/api/python/stable/reference/api/polars.datatypes.Enum.html#polars.datatypes.Enum
_POLARS_TYPE_DENYLIST = set([pl.Enum, pl.Object, pl.Unknown])


def _get_available_port() -> int:
    """Identify the first available port out of all GRPC_PORTS"""
    for port in GRPC_PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
            except Exception:
                continue
        return port

    raise ValueError(f'None of the expected ports {GRPC_PORTS} are available.')


def _serialize(data: Any) -> kaggle_evaluation_proto.Payload:
    """Maps input data of one of several allow-listed types to a protobuf message to be sent over gRPC.

    Args:
        data: The input data to be mapped. Any of the types listed below are accepted.

    Returns:
        The Payload protobuf message.

    Raises:
        TypeError if data is of an unsupported type.
    """
    # Python primitives and Numpy scalars
    if isinstance(data, np.generic):
        # Numpy functions that return a single number return numpy scalars instead of python primitives.
        # In some cases this difference matters: https://numpy.org/devdocs/release/2.0.0-notes.html#representation-of-numpy-scalars-changed
        # Ex: np.mean(1,2) yields np.float64(1.5) instead of 1.5.
        # Check for numpy scalars first since most of them also inherit from python primitives.
        # For example, `np.float64(1.5)` is an instance of `float` among many other things.
        # https://numpy.org/doc/stable/reference/arrays.scalars.html
        assert data.shape == ()  # Additional validation that the np.generic type remains solely for scalars
        assert isinstance(data, np.number) or isinstance(data, np.bool_)  # No support for bytes, strings, objects, etc
        buffer = io.BytesIO()
        np.save(buffer, data, allow_pickle=False)
        return kaggle_evaluation_proto.Payload(numpy_scalar_value=buffer.getvalue())
    elif isinstance(data, str):
        return kaggle_evaluation_proto.Payload(str_value=data)
    elif isinstance(data, bool):  # bool is a subclass of int, so check that first
        return kaggle_evaluation_proto.Payload(bool_value=data)
    elif isinstance(data, int):
        return kaggle_evaluation_proto.Payload(int_value=data)
    elif isinstance(data, float):
        return kaggle_evaluation_proto.Payload(float_value=data)
    elif data is None:
        return kaggle_evaluation_proto.Payload(none_value=True)
    # Iterables for nested types
    if isinstance(data, list):
        return kaggle_evaluation_proto.Payload(list_value=kaggle_evaluation_proto.PayloadList(payloads=map(_serialize, data)))
    elif isinstance(data, tuple):
        return kaggle_evaluation_proto.Payload(tuple_value=kaggle_evaluation_proto.PayloadList(payloads=map(_serialize, data)))
    elif isinstance(data, dict):
        serialized_dict = {}
        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError(f'KaggleEvaluation only supports dicts with keys of type str, found {type(key)}.')
            serialized_dict[key] = _serialize(value)
        return kaggle_evaluation_proto.Payload(dict_value=kaggle_evaluation_proto.PayloadMap(payload_map=serialized_dict))
    # Allowlisted special types
    if isinstance(data, pd.DataFrame):
        buffer = io.BytesIO()
        data.to_parquet(buffer, index=False, compression='lz4')
        return kaggle_evaluation_proto.Payload(pandas_dataframe_value=buffer.getvalue())
    elif isinstance(data, pl.DataFrame):
        data_types = set(i.base_type() for i in data.dtypes)
        banned_types = _POLARS_TYPE_DENYLIST.intersection(data_types)
        if len(banned_types) > 0:
            raise TypeError(f'Unsupported Polars data type(s): {banned_types}')

        table = data.to_arrow()
        buffer = io.BytesIO()
        with pyarrow.ipc.new_stream(buffer, table.schema, options=pyarrow.ipc.IpcWriteOptions(compression='lz4')) as writer:
            writer.write_table(table)
        return kaggle_evaluation_proto.Payload(polars_dataframe_value=buffer.getvalue())
    elif isinstance(data, pd.Series):
        buffer = io.BytesIO()
        # Can't serialize a pd.Series directly to parquet, must use intermediate DataFrame
        pd.DataFrame(data).to_parquet(buffer, index=False, compression='lz4')
        return kaggle_evaluation_proto.Payload(pandas_series_value=buffer.getvalue())
    elif isinstance(data, pl.Series):
        buffer = io.BytesIO()
        # Can't serialize a pl.Series directly to parquet, must use intermediate DataFrame
        pl.DataFrame(data).write_parquet(buffer, compression='lz4', statistics=False)
        return kaggle_evaluation_proto.Payload(polars_series_value=buffer.getvalue())
    elif isinstance(data, np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, data, allow_pickle=False)
        return kaggle_evaluation_proto.Payload(numpy_array_value=buffer.getvalue())
    elif isinstance(data, io.BytesIO):
        return kaggle_evaluation_proto.Payload(bytes_io_value=data.getvalue())

    raise TypeError(f'Type {type(data)} not supported for KaggleEvaluation.')


def _deserialize(payload: kaggle_evaluation_proto.Payload) -> Any:
    """Maps a Payload protobuf message to a value of whichever type was set on the message.

    Args:
        payload: The message to be mapped.

    Returns:
        A value of one of several allow-listed types.

    Raises:
        TypeError if an unexpected value data type is found.
    """
    # Primitives
    if payload.WhichOneof('value') == 'str_value':
        return payload.str_value
    elif payload.WhichOneof('value') == 'bool_value':
        return payload.bool_value
    elif payload.WhichOneof('value') == 'int_value':
        return payload.int_value
    elif payload.WhichOneof('value') == 'float_value':
        return payload.float_value
    elif payload.WhichOneof('value') == 'none_value':
        return None
    # Iterables for nested types
    elif payload.WhichOneof('value') == 'list_value':
        return list(map(_deserialize, payload.list_value.payloads))
    elif payload.WhichOneof('value') == 'tuple_value':
        return tuple(map(_deserialize, payload.tuple_value.payloads))
    elif payload.WhichOneof('value') == 'dict_value':
        return {key: _deserialize(value) for key, value in payload.dict_value.payload_map.items()}
    # Allowlisted special types
    elif payload.WhichOneof('value') == 'pandas_dataframe_value':
        return pd.read_parquet(io.BytesIO(payload.pandas_dataframe_value))
    elif payload.WhichOneof('value') == 'polars_dataframe_value':
        with pyarrow.ipc.open_stream(payload.polars_dataframe_value) as reader:
            table = reader.read_all()
        return pl.from_arrow(table)
    elif payload.WhichOneof('value') == 'pandas_series_value':
        # Pandas will still read a single column csv as a DataFrame.
        df = pd.read_parquet(io.BytesIO(payload.pandas_series_value))
        return pd.Series(df[df.columns[0]])
    elif payload.WhichOneof('value') == 'polars_series_value':
        return pl.Series(pl.read_parquet(io.BytesIO(payload.polars_series_value)))
    elif payload.WhichOneof('value') == 'numpy_array_value':
        return np.load(io.BytesIO(payload.numpy_array_value), allow_pickle=False)
    elif payload.WhichOneof('value') == 'numpy_scalar_value':
        data = np.load(io.BytesIO(payload.numpy_scalar_value), allow_pickle=False)
        # As of Numpy 2.0.2, np.load for a numpy scalar yields a dimensionless array instead of a scalar
        data = data.dtype.type(data)  # Restore the expected numpy scalar type.
        assert data.shape == ()  # Additional validation that the np.generic type remains solely for scalars
        assert isinstance(data, np.number) or isinstance(data, np.bool_)  # No support for bytes, strings, objects, etc
        return data
    elif payload.WhichOneof('value') == 'bytes_io_value':
        return io.BytesIO(payload.bytes_io_value)

    raise TypeError(f'Found unknown Payload case {payload.WhichOneof("value")}')


### Client code


class Client:
    """
    Class which allows callers to make KaggleEvaluation requests.
    """

    def __init__(self, channel_address: str = 'localhost') -> None:
        self.channel_address = channel_address
        self.channel: Optional[grpc.Channel] = None
        self._made_first_connection = False
        self.endpoint_deadline_seconds = DEFAULT_DEADLINE_SECONDS
        self.stub: Optional[kaggle_evaluation_grpc.KaggleEvaluationServiceStub] = None

    def _send_with_deadline(self, request) -> kaggle_evaluation_proto.KaggleEvaluationResponse:
        """Sends a message to the server while also:
        - Throwing an error as soon as the inference_server container has been shut down.
        - Setting a deadline of STARTUP_LIMIT_SECONDS for the inference_server to startup.
        """
        if self._made_first_connection:
            try:
                return self.stub.Send(request, wait_for_ready=False, timeout=self.endpoint_deadline_seconds)
            except _InactiveRpcError as err:
                if 'StatusCode.DEADLINE_EXCEEDED' in str(err):
                    raise GRPCDeadlineError()
                else:
                    raise err
            except Exception as err:
                raise err

        first_call_time = time.time()
        # Allow time for the server to start as long as its container is running
        while time.time() - first_call_time < STARTUP_LIMIT_SECONDS:
            for port in GRPC_PORTS:
                self.channel = grpc.insecure_channel(f'{self.channel_address}:{port}', options=_GRPC_CHANNEL_OPTIONS)
                self.stub = kaggle_evaluation_grpc.KaggleEvaluationServiceStub(self.channel)
                try:
                    response = self.stub.Send(request, wait_for_ready=False)
                    self._made_first_connection = True
                    return response
                except grpc._channel._InactiveRpcError as err:
                    if 'StatusCode.UNAVAILABLE' not in str(err):
                        raise err
                # Confirm the inference_server container is still alive & it's worth waiting on the server.
                # If the inference_server container is no longer running this will throw a socket.gaierror.
                socket.gethostbyname(self.channel_address)
                time.sleep(_RETRY_SLEEP_SECONDS)

        if not self._made_first_connection:
            raise RuntimeError(f'Failed to connect to server after waiting {STARTUP_LIMIT_SECONDS} seconds')

    def serialize_request(self, name: str, *args, **kwargs) -> kaggle_evaluation_proto.KaggleEvaluationRequest:
        """Serialize a single request. Exists as a separate function from `send`
        to enable gateway concurrency for some competitions.
        """
        already_serialized = (len(args) == 1) and isinstance(args[0], kaggle_evaluation_proto.KaggleEvaluationRequest)
        if already_serialized:
            return args[0]  # args is a tuple of length 1 containing the request
        return kaggle_evaluation_proto.KaggleEvaluationRequest(
            name=name, args=map(_serialize, args), kwargs={key: _serialize(value) for key, value in kwargs.items()}
        )

    def send(self, name: str, *args, **kwargs) -> Any:
        """Sends a single KaggleEvaluation request.

        Args:
            name: The endpoint name for the request.
            *args: Variable-length/type arguments to be supplied on the request.
            **kwargs: Key-value arguments to be supplied on the request.

        Returns:
            The response, which is of one of several allow-listed data types.
        """
        request = self.serialize_request(name, *args, **kwargs)
        response = self._send_with_deadline(request)
        return _deserialize(response.payload)

    def close(self) -> None:
        if self.channel is not None:
            self.channel.close()


### Server code


class KaggleEvaluationServiceServicer(kaggle_evaluation_grpc.KaggleEvaluationServiceServicer):
    """
    Class which allows serving responses to KaggleEvaluation requests. The inference_server will run this service to listen for and respond
    to requests from the Gateway. The Gateway may also listen for requests from the inference_server in some cases.
    """

    def __init__(self, listeners: Tuple[Callable]):
        self.listeners_map = dict((func.__name__, func) for func in listeners)

    # pylint: disable=unused-argument
    def Send(
        self, request: kaggle_evaluation_proto.KaggleEvaluationRequest, context: grpc.ServicerContext
    ) -> kaggle_evaluation_proto.KaggleEvaluationResponse:
        """Handler for gRPC requests that deserializes arguments, calls a user-registered function for handling the
        requested endpoint, then serializes and returns the response.

        Args:
            request: The KaggleEvaluationRequest protobuf message.
            context: (Unused) gRPC context.

        Returns:
            The KaggleEvaluationResponse protobuf message.

        Raises:
            NotImplementedError if the caller has not registered a handler for the requested endpoint.
        """
        if request.name not in self.listeners_map:
            raise NotImplementedError(f'No listener for {request.name} was registered.')

        args = map(_deserialize, request.args)
        kwargs = {key: _deserialize(value) for key, value in request.kwargs.items()}
        response_function = self.listeners_map[request.name]
        response_payload = _serialize(response_function(*args, **kwargs))
        return kaggle_evaluation_proto.KaggleEvaluationResponse(payload=response_payload)


def define_server(*endpoint_listeners: Callable) -> grpc.server:
    """Registers the endpoints that the container is able to respond to, then starts a server which listens for
    those endpoints. The endpoints that need to be implemented will depend on the specific competition.

    Args:
        endpoint_listeners: Tuple of functions that define how requests to the endpoint of the function name should be
            handled.

    Returns:
        The gRPC server object, which has been started. It should be stopped at exit time.

    Raises:
        ValueError if parameter values are invalid.
    """
    if not endpoint_listeners:
        raise ValueError('Must pass at least one endpoint listener, e.g. `predict`')
    for func in endpoint_listeners:
        if not isinstance(func, Callable):
            raise ValueError(f'Endpoint listeners passed to `serve` must be functions, got {type(func)}')
        if func.__name__ == '<lambda>':
            raise ValueError('Functions passed as endpoint listeners must be named')

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=_GRPC_CHANNEL_OPTIONS)
    kaggle_evaluation_grpc.add_KaggleEvaluationServiceServicer_to_server(KaggleEvaluationServiceServicer(endpoint_listeners), server)
    grpc_port = _get_available_port()
    server.add_insecure_port(f'[::]:{grpc_port}')
    return server
```


`kaggle_evaluation` k ander subfolder `core` k ander ye **templates.py** file ka full code
```python
"""Template for the two classes hosts should customize for each competition."""

import abc
import os
import time
import sys
import traceback
import warnings

from typing import Callable, Generator, Optional, Tuple, Any, List

import kaggle_evaluation.core.base_gateway
import kaggle_evaluation.core.relay


_initial_import_time = time.time()
_issued_startup_time_warning = False


class Gateway(kaggle_evaluation.core.base_gateway.BaseGateway, abc.ABC):
    """
    Template to start with when writing a new gateway.
    In most cases, hosts should only need to write get_all_predictions.
    There are two main methods for sending data to the inference_server hosts should understand:
    - Small datasets: use `self.predict`. Competitors will receive the data passed to self.predict as
    Python objects in memory. This is just a wrapper for self.client.send(); you can write additional
    wrappers if necessary.
    - Large datasets: it's much faster to send data via self.share_files, which is equivalent to making
    files available via symlink. See base_gateway.BaseGateway.share_files for the full details.
    """

    @abc.abstractmethod
    def unpack_data_paths(self) -> None:
        """Map the contents of self.data_paths to the competition-specific entries
        Each competition should respect these paths to make it easy for competitors to
        run tests on their local machines or with custom files.

        Should include default paths to support data_paths = None.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_data_batches(self) -> Generator:
        """Used by the default implementation of `get_all_predictions` so we can
        ensure `validate_prediction_batch` is run every time `predict` is called.

        This method must yield both the batch of data to be sent to `predict` and a series
        of row IDs to be sent to `validate_prediction_batch`.
        """
        raise NotImplementedError

    def get_all_predictions(self) -> Tuple[List[Any], List[Any]]:
        all_predictions = []
        all_row_ids = []
        for data_batch, row_ids in self.generate_data_batches():
            predictions = self.predict(*data_batch)
            self.validate_prediction_batch(predictions, row_ids)
            all_predictions.append(predictions)
            all_row_ids.append(row_ids)
        return all_predictions, all_row_ids

    def predict(self, *args, **kwargs) -> Any:
        """self.predict will send all data in args and kwargs to the user container, and
        instruct the user container to generate a `predict` response.

        Returns:
            Any: The prediction from the user container.
        """
        try:
            return self.client.send('predict', *args, **kwargs)
        except Exception as e:
            self.handle_server_error(e, 'predict')

    def set_response_timeout_seconds(self, timeout_seconds: int) -> None:
        # Also store timeout_seconds in an easy place for for competitor to access.
        self.timeout_seconds = timeout_seconds
        # Set a response deadline that will apply after the very first repsonse
        self.client.endpoint_deadline_seconds = timeout_seconds

    def run(self) -> None:
        error = None
        try:
            self.unpack_data_paths()
            predictions, row_ids = self.get_all_predictions()
            self.write_submission(predictions, row_ids)
        except kaggle_evaluation.core.base_gateway.GatewayRuntimeError as gre:
            error = gre
        except Exception:
            # Get the full stack trace
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

            error = kaggle_evaluation.core.base_gateway.GatewayRuntimeError(
                kaggle_evaluation.core.base_gateway.GatewayRuntimeErrorType.GATEWAY_RAISED_EXCEPTION, error_str
            )

        self.client.close()
        if self.server:
            self.server.stop(0)

        if kaggle_evaluation.core.base_gateway.IS_RERUN:
            self.write_result(error)
        elif error:
            # For local testing
            raise error


class InferenceServer(abc.ABC):
    """
    Base class for competition participants to inherit from when writing their submission. In most cases, users should
    only need to implement a `predict` function or other endpoints to pass to this class's constructor, and hosts will
    provide a mock Gateway for testing.
    """

    def __init__(self, *endpoint_listeners: Callable):
        self.server = kaggle_evaluation.core.relay.define_server(*endpoint_listeners)
        self.client = None  # The inference_server can have a client but it isn't typically necessary.
        self._issued_startup_time_warning = False
        self._startup_limit_seconds = kaggle_evaluation.core.relay.STARTUP_LIMIT_SECONDS

    def serve(self) -> None:
        self.server.start()
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None:
            self.server.wait_for_termination()  # This will block all other code

    @abc.abstractmethod
    def _get_gateway_for_test(self, data_paths, file_share_dir=None, *args, **kwargs):
        # Must return a version of the competition-specific gateway able to load data for unit tests.
        raise NotImplementedError

    def run_local_gateway(self, data_paths: Optional[Tuple[str]] = None, file_share_dir: str = None, *args, **kwargs) -> None:
        """Construct a copy of the gateway that uses local file paths."""
        global _issued_startup_time_warning
        script_elapsed_seconds = time.time() - _initial_import_time
        if script_elapsed_seconds > self._startup_limit_seconds and not _issued_startup_time_warning:
            warnings.warn(
                f"""{int(script_elapsed_seconds)} seconds elapsed before server startup.
                This exceeds the startup time limit of {int(self._startup_limit_seconds)} seconds that the gateway will enforce
                during the rerun on the hidden test set. Start the server before performing any time consuming steps.""",
                category=RuntimeWarning,
            )
            _issued_startup_time_warning = True

        self.server.start()
        try:
            self.gateway = self._get_gateway_for_test(data_paths, file_share_dir, *args, **kwargs)
            self.gateway.run()
        except Exception as err:
            raise err from None
        finally:
            self.server.stop(0)
```


`kaggle_evaluation` k ander subfolder `core` k ander subfolder `generated` k ander ye **__init__.py** file ka full code
```python
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

`kaggle_evaluation` k ander subfolder `core` k ander subfolder `generated` k ander ye **kaggle_evaluation_pb2.py** file ka full code
```python
# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: kaggle_evaluation.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17kaggle_evaluation.proto\x12\x18kaggle_evaluation_client\"\xf9\x01\n\x17KaggleEvaluationRequest\x12\x0c\n\x04name\x18\x01 \x01(\t\x12/\n\x04\x61rgs\x18\x02 \x03(\x0b\x32!.kaggle_evaluation_client.Payload\x12M\n\x06kwargs\x18\x03 \x03(\x0b\x32=.kaggle_evaluation_client.KaggleEvaluationRequest.KwargsEntry\x1aP\n\x0bKwargsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x30\n\x05value\x18\x02 \x01(\x0b\x32!.kaggle_evaluation_client.Payload:\x02\x38\x01\"N\n\x18KaggleEvaluationResponse\x12\x32\n\x07payload\x18\x01 \x01(\x0b\x32!.kaggle_evaluation_client.Payload\"\x8d\x04\n\x07Payload\x12\x13\n\tstr_value\x18\x01 \x01(\tH\x00\x12\x14\n\nbool_value\x18\x02 \x01(\x08H\x00\x12\x13\n\tint_value\x18\x03 \x01(\x12H\x00\x12\x15\n\x0b\x66loat_value\x18\x04 \x01(\x02H\x00\x12\x14\n\nnone_value\x18\x05 \x01(\x08H\x00\x12;\n\nlist_value\x18\x06 \x01(\x0b\x32%.kaggle_evaluation_client.PayloadListH\x00\x12<\n\x0btuple_value\x18\x07 \x01(\x0b\x32%.kaggle_evaluation_client.PayloadListH\x00\x12:\n\ndict_value\x18\x08 \x01(\x0b\x32$.kaggle_evaluation_client.PayloadMapH\x00\x12 \n\x16pandas_dataframe_value\x18\t \x01(\x0cH\x00\x12 \n\x16polars_dataframe_value\x18\n \x01(\x0cH\x00\x12\x1d\n\x13pandas_series_value\x18\x0b \x01(\x0cH\x00\x12\x1d\n\x13polars_series_value\x18\x0c \x01(\x0cH\x00\x12\x1b\n\x11numpy_array_value\x18\r \x01(\x0cH\x00\x12\x1c\n\x12numpy_scalar_value\x18\x0e \x01(\x0cH\x00\x12\x18\n\x0e\x62ytes_io_value\x18\x0f \x01(\x0cH\x00\x42\x07\n\x05value\"B\n\x0bPayloadList\x12\x33\n\x08payloads\x18\x01 \x03(\x0b\x32!.kaggle_evaluation_client.Payload\"\xad\x01\n\nPayloadMap\x12I\n\x0bpayload_map\x18\x01 \x03(\x0b\x32\x34.kaggle_evaluation_client.PayloadMap.PayloadMapEntry\x1aT\n\x0fPayloadMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x30\n\x05value\x18\x02 \x01(\x0b\x32!.kaggle_evaluation_client.Payload:\x02\x38\x01\x32\x8a\x01\n\x17KaggleEvaluationService\x12o\n\x04Send\x12\x31.kaggle_evaluation_client.KaggleEvaluationRequest\x1a\x32.kaggle_evaluation_client.KaggleEvaluationResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'kaggle_evaluation_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_KAGGLEEVALUATIONREQUEST_KWARGSENTRY']._options = None
  _globals['_KAGGLEEVALUATIONREQUEST_KWARGSENTRY']._serialized_options = b'8\001'
  _globals['_PAYLOADMAP_PAYLOADMAPENTRY']._options = None
  _globals['_PAYLOADMAP_PAYLOADMAPENTRY']._serialized_options = b'8\001'
  _globals['_KAGGLEEVALUATIONREQUEST']._serialized_start=54
  _globals['_KAGGLEEVALUATIONREQUEST']._serialized_end=303
  _globals['_KAGGLEEVALUATIONREQUEST_KWARGSENTRY']._serialized_start=223
  _globals['_KAGGLEEVALUATIONREQUEST_KWARGSENTRY']._serialized_end=303
  _globals['_KAGGLEEVALUATIONRESPONSE']._serialized_start=305
  _globals['_KAGGLEEVALUATIONRESPONSE']._serialized_end=383
  _globals['_PAYLOAD']._serialized_start=386
  _globals['_PAYLOAD']._serialized_end=911
  _globals['_PAYLOADLIST']._serialized_start=913
  _globals['_PAYLOADLIST']._serialized_end=979
  _globals['_PAYLOADMAP']._serialized_start=982
  _globals['_PAYLOADMAP']._serialized_end=1155
  _globals['_PAYLOADMAP_PAYLOADMAPENTRY']._serialized_start=1071
  _globals['_PAYLOADMAP_PAYLOADMAPENTRY']._serialized_end=1155
  _globals['_KAGGLEEVALUATIONSERVICE']._serialized_start=1158
  _globals['_KAGGLEEVALUATIONSERVICE']._serialized_end=1296
# @@protoc_insertion_point(module_scope)
```

`kaggle_evaluation` k ander subfolder `core` k ander subfolder `generated` k ander ye **kaggle_evaluation_pb2_grpc.py** file ka full code
```python
# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import kaggle_evaluation_pb2 as kaggle__evaluation__pb2


class KaggleEvaluationServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Send = channel.unary_unary(
                '/kaggle_evaluation_client.KaggleEvaluationService/Send',
                request_serializer=kaggle__evaluation__pb2.KaggleEvaluationRequest.SerializeToString,
                response_deserializer=kaggle__evaluation__pb2.KaggleEvaluationResponse.FromString,
                )


class KaggleEvaluationServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Send(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_KaggleEvaluationServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Send': grpc.unary_unary_rpc_method_handler(
                    servicer.Send,
                    request_deserializer=kaggle__evaluation__pb2.KaggleEvaluationRequest.FromString,
                    response_serializer=kaggle__evaluation__pb2.KaggleEvaluationResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'kaggle_evaluation_client.KaggleEvaluationService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class KaggleEvaluationService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Send(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/kaggle_evaluation_client.KaggleEvaluationService/Send',
            kaggle__evaluation__pb2.KaggleEvaluationRequest.SerializeToString,
            kaggle__evaluation__pb2.KaggleEvaluationResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
```



*Summary*
* 15 files
    * .csv 4
    * .py 10
    * .proto 1

* 693 columns
    * Decimal 668
    * String 13
    * Integer 12



## Competition Rules

**ENTRY IN THIS COMPETITION CONSTITUTES YOUR ACCEPTANCE OF THESE OFFICIAL COMPETITION RULES.
See Section 3.18 for defined terms**

The Competition named below is a skills-based competition to promote and further the field of data science. You must register via the Competition Website to enter. To enter the Competition, you must agree to these Official Competition Rules, which incorporate by reference the provisions and content of the Competition Website and any Specific Competition Rules herein (collectively, the "Rules"). Please read these Rules carefully before entry to ensure you understand and agree. You further agree that Submission in the Competition constitutes agreement to these Rules. You may not submit to the Competition and are not eligible to receive the prizes associated with this Competition unless you agree to these Rules. These Rules form a binding legal agreement between you and the Competition Sponsor with respect to the Competition. Your competition Submissions must conform to the requirements stated on the Competition Website. Your Submissions will be scored based on the evaluation metric described on the Competition Website. Subject to compliance with the Competition Rules, Prizes, if any, will be awarded to Participants with the best scores, based on the merits of the data science models submitted. See below for the complete Competition Rules. For Competitions designated as hackathons by the Competition Sponsor (“Hackathons”), your Submissions will be judged by the Competition Sponsor based on the evaluation rubric set forth on the Competition Website (“Evaluation Rubric”). The Prizes, if any, will be awarded to Participants with the highest ranking(s) as determined by the Competition Sponsor based on such rubric.

You cannot sign up to Kaggle from multiple accounts and therefore you cannot enter or submit from multiple accounts.

1. COMPETITION-SPECIFIC TERMS
1. COMPETITION TITLE
CMI - Detect Behavior with Sensor Data

2. COMPETITION SPONSOR
Child Mind Institute Inc.

3. COMPETITION SPONSOR ADDRESS
215 E 50TH St New York, NY, 10022-7701

4. COMPETITION WEBSITE
https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data

5. TOTAL PRIZES AVAILABLE: $ 50,000
1st Place - $ 15,000
2nd Place - $ 10,000
3rd Place - $ 8,000
4th Place - $ 7,000
5th Place - $ 5,000
6th Place - $ 5,000
6. WINNER LICENSE TYPE
OPEN SOURCE - LGPL 2.1

7. DATA ACCESS AND USE
CC BY-NC-SA 4.0 Non-Commercial Use, for the purposes of the competition. Re-distribution or re-identification of any data is strictly prohibited.

2. COMPETITION-SPECIFIC RULES
In addition to the provisions of the General Competition Rules below, you understand and agree to these Competition-Specific Rules required by the Competition Sponsor:

1. TEAM LIMITS
a. The maximum Team size is five (5).
b. Team mergers are allowed and can be performed by the Team leader. In order to merge, the combined Team must have a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline. The maximum allowed is the number of Submissions per day multiplied by the number of days the competition has been running. For Hackathons, each team is allowed one (1) Submission; any Submissions submitted by Participants before merging into a Team will be unsubmitted.

2. SUBMISSION LIMITS
a. You may submit a maximum of five (5) Submissions per day.
b. You may select up to two (2) Final Submissions for judging.
c. For Hackathons, each Team may submit one (1) Submission only.

3. COMPETITION TIMELINE
a. Competition Timeline dates (including Entry Deadline, Final Submission Deadline, Start Date, and Team Merger Deadline, as applicable) are reflected on the competition’s Overview > Timeline page.

4. COMPETITION DATA
a. Data Access and Use.

You may access and use the Competition Data for non-commercial purposes only, including for participating in the Competition and on Kaggle.com forums, and for academic research and education. The Competition Sponsor reserves the right to disqualify any Participant who uses the Competition Data other than as permitted by the Competition Website and these Rules.

Phenotypic (e.g., tabular, survey) data used in this Challenge have been contributed in part by the Healthy Brain Network study and are de-identified patient data, meaning that the contributing institutions have taken reasonable care to remove from them all personally identifiable information. As a participant in the Competition, you agree 1) not to make copies of, or in any way redistribute, any of the data made available to you, 2) not to attempt to re-identify any personal information based on the data, 3) not to attempt to probe the test set's labels, and 4) to notify the Competition organizers of any personally identifiable information you encounter in the data by posting a message to the Kaggle community discussion forums.

No data sources external to those directly shared as a part of this competition that are either copies of, or derived from, the Healthy Brain Network dataset may be used in this competition; such use will result in automatic disqualification from eligibility. Participants with access to the HBN dataset via a Data Usage Agreement (DUA) with the Child Mind Institute may not use any data extracted from this research data for this competition, and participants who are not active DUA holders agree that they will not request access to the dataset for use in unrelated projects for the duration of this competition.

If you reference or use the dataset in any form, include the following citation:
“CMI 2025 Detect Behavior with Sensor Data Challenge”

b. Data Security.

You agree to use reasonable and suitable measures to prevent persons who have not formally agreed to these Rules from gaining access to the Competition Data. You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition. You agree to notify Kaggle immediately upon learning of any possible unauthorized transmission of or unauthorized access to the Competition Data and agree to work with Kaggle to rectify any unauthorized transmission or access.
5. WINNER LICENSE
a. Under Section 2.8 (Winners Obligations) of the General Rules below, you hereby grant and will grant the Competition Sponsor the following license(s) with respect to your Submission if you are a Competition winner:

Open Source: You hereby license and will license your winning Submission and the source code used to generate the Submission under an Open Source Initiative-approved license (see LGPL 2.1) that in no event limits commercial use of such code or model containing or depending on such code.
b. You may be required by the Sponsor to provide a detailed description of how the winning Submission was generated, to the Competition Sponsor’s specifications, as outlined in Section 2.8, Winner’s Obligations. This may include a detailed description of methodology, where one must be able to reproduce the approach by reading the description, and includes a detailed explanation of the architecture, preprocessing, loss function, training details, hyper-parameters, etc. The description should also include a link to a code repository with complete and detailed instructions so that the results obtained can be reproduced.

6. EXTERNAL DATA AND TOOLS
a. You may use data other than the Competition Data (“External Data”) to develop and test your Submissions. However, you will ensure the External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants, or satisfies the Reasonableness criteria as outlined in Section 2.6.b below. The ability to use External Data under this Section does not limit your other obligations under these Competition Rules, including but not limited to Section 2.8 (Winners Obligations).

b. The use of external data and models is acceptable unless specifically prohibited by the Host. Because of the potential costs or restrictions (e.g., “geo restrictions”) associated with obtaining rights to use external data or certain software and associated tools, their use must be “reasonably accessible to all” and of “minimal cost”. Also, regardless of the cost challenges as they might affect all Participants during the course of the competition, the costs of potentially procuring a license for software used to generate a Submission, must also be considered. The Host will employ an assessment of whether or not the following criteria can exclude the use of the particular LLM, data set(s), or tool(s):

Are Participants being excluded from a competition because of the "excessive" costs for access to certain LLMs, external data, or tools that might be used by other Participants. The Host will assess the excessive cost concern by applying a “Reasonableness” standard (the “Reasonableness Standard”). The Reasonableness Standard will be determined and applied by the Host in light of things like cost thresholds and accessibility.

By way of example only, a small subscription charge to use additional elements of a large language model such as Gemini Advanced are acceptable if meeting the Reasonableness Standard of Sec. 8.2. Purchasing a license to use a proprietary dataset that exceeds the cost of a prize in the competition would not be considered reasonable.

c. Automated Machine Learning Tools (“AMLT”)

Individual Participants and Teams may use automated machine learning tool(s) (“AMLT”) (e.g., Google toML, H2O Driverless AI, etc.) to create a Submission, provided that the Participant or Team ensures that they have an appropriate license to the AMLT such that they are able to comply with the Competition Rules.
7. ELIGIBILITY
a. Unless otherwise stated in the Competition-Specific Rules above or prohibited by internal policies of the Competition Entities, employees, interns, contractors, officers and directors of Competition Entities may enter and participate in the Competition, but are not eligible to win any Prizes. "Competition Entities" means the Competition Sponsor, Kaggle Inc., and their respective parent companies, subsidiaries and affiliates. If you are such a Participant from a Competition Entity, you are subject to all applicable internal policies of your employer with respect to your participation.

8. WINNER’S OBLIGATIONS
a. As a condition to being awarded a Prize, a Prize winner must fulfill the following obligations:

Deliver to the Competition Sponsor the final model's software code as used to generate the winning Submission and associated documentation. The delivered software code should follow these documentation guidelines, must be capable of generating the winning Submission, and contain a description of resources required to build and/or run the executable code successfully. For avoidance of doubt, delivered software code should include training code, inference code, and a description of the required computational environment. For Hackathons, the Submission deliverables will be as described on the Competition Website, which may be information or materials that are not software code.
a. To the extent that the final model’s software code includes generally commercially available software that is not owned by you, but that can be procured by the Competition Sponsor without undue expense, then instead of delivering the code for that software to the Competition Sponsor, you must identify that software, method for procuring it, and any parameters or other information necessary to replicate the winning Submission; Individual Participants and Teams who create a Submission using an AMLT may win a Prize. However, for clarity, the potential winner’s Submission must still meet the requirements of these Rules, including but not limited to Section 2.5 (Winners License), Section 2.8 (Winners Obligations), and Section 3.14 (Warranty, Indemnity, and Release).”

b. Individual Participants and Teams who create a Submission using an AMLT may win a Prize. However, for clarity, the potential winner’s Submission must still meet the requirements of these Rules,

Grant to the Competition Sponsor the license to the winning Submission stated in the Competition Specific Rules above, and represent that you have the unrestricted right to grant that license;

Sign and return all Prize acceptance documents as may be required by Competition Sponsor or Kaggle, including without limitation: (a) eligibility certifications; (b) licenses, releases and other agreements required under the Rules; and (c) U.S. tax forms (such as IRS Form W-9 if U.S. resident, IRS Form W-8BEN if foreign resident, or future equivalents).

9. GOVERNING LAW
a. Unless otherwise provided in the Competition Specific Rules above, all claims arising out of or relating to these Rules will be governed by New York law, excluding its conflict of laws rules, and will be litigated exclusively in the Federal or State courts of New York County, New York, USA. The parties consent to personal jurisdiction in those courts. If any provision of these Rules is held to be invalid or unenforceable, all remaining provisions of the Rules will remain in full force and effect.

3. GENERAL COMPETITION RULES - BINDING AGREEMENT
1. ELIGIBILITY
a. To be eligible to enter the Competition, you must be:

a registered account holder at Kaggle.com;
the older of 18 years old or the age of majority in your jurisdiction of residence (unless otherwise agreed to by Competition Sponsor and appropriate parental/guardian consents have been obtained by Competition Sponsor);
not a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, Syria, or North Korea; and
not a person or representative of an entity under U.S. export controls or sanctions (see: https://www.treasury.gov/resourcecenter/sanctions/Programs/Pages/Programs.aspx).
b. Competitions are open to residents of the United States and worldwide, except that if you are a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, Syria, North Korea, or are subject to U.S. export controls or sanctions, you may not enter the Competition. Other local rules and regulations may apply to you, so please check your local laws to ensure that you are eligible to participate in skills-based competitions. The Competition Host reserves the right to forego or award alternative Prizes where needed to comply with local laws. If a winner is located in a country where prizes cannot be awarded, then they are not eligible to receive a prize.

c. If you are entering as a representative of a company, educational institution or other legal entity, or on behalf of your employer, these rules are binding on you, individually, and the entity you represent or where you are an employee. If you are acting within the scope of your employment, or as an agent of another party, you warrant that such party or your employer has full knowledge of your actions and has consented thereto, including your potential receipt of a Prize. You further warrant that your actions do not violate your employer's or entity's policies and procedures.

d. The Competition Sponsor reserves the right to verify eligibility and to adjudicate on any dispute at any time. If you provide any false information relating to the Competition concerning your identity, residency, mailing address, telephone number, email address, ownership of right, or information required for entering the Competition, you may be immediately disqualified from the Competition.

2. SPONSOR AND HOSTING PLATFORM
a. The Competition is sponsored by Competition Sponsor named above. The Competition is hosted on behalf of Competition Sponsor by Kaggle Inc. ("Kaggle"). Kaggle is an independent contractor of Competition Sponsor, and is not a party to this or any agreement between you and Competition Sponsor. You understand that Kaggle has no responsibility with respect to selecting the potential Competition winner(s) or awarding any Prizes. Kaggle will perform certain administrative functions relating to hosting the Competition, and you agree to abide by the provisions relating to Kaggle under these Rules. As a Kaggle.com account holder and user of the Kaggle competition platform, remember you have accepted and are subject to the Kaggle Terms of Service at www.kaggle.com/terms in addition to these Rules.

3. COMPETITION PERIOD
a. For the purposes of Prizes, the Competition will run from the Start Date and time to the Final Submission Deadline (such duration the “Competition Period”). The Competition Timeline is subject to change, and Competition Sponsor may introduce additional hurdle deadlines during the Competition Period. Any updated or additional deadlines will be publicized on the Competition Website. It is your responsibility to check the Competition Website regularly to stay informed of any deadline changes. YOU ARE RESPONSIBLE FOR DETERMINING THE CORRESPONDING TIME ZONE IN YOUR LOCATION.

4. COMPETITION ENTRY
a. NO PURCHASE NECESSARY TO ENTER OR WIN. To enter the Competition, you must register on the Competition Website prior to the Entry Deadline, and follow the instructions for developing and entering your Submission through the Competition Website. Your Submissions must be made in the manner and format, and in compliance with all other requirements, stated on the Competition Website (the "Requirements"). Submissions must be received before any Submission deadlines stated on the Competition Website. Submissions not received by the stated deadlines will not be eligible to receive a Prize.
b. Except as expressly allowed in Hackathons as set forth on the Competition Website, submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records.
c. If the Competition is a multi-stage competition with temporally separate training and/or test data, one or more valid Submissions may be required during each Competition stage in the manner described on the Competition Website in order for the Submissions to be Prize eligible.
d. Submissions are void if they are in whole or part illegible, incomplete, damaged, altered, counterfeit, obtained through fraud, or late. Competition Sponsor reserves the right to disqualify any entrant who does not follow these Rules, including making a Submission that does not meet the Requirements.

5. INDIVIDUALS AND TEAMS
a. Individual Account. You may make Submissions only under one, unique Kaggle.com account. You will be disqualified if you make Submissions through more than one Kaggle account, or attempt to falsify an account to act as your proxy. You may submit up to the maximum number of Submissions per day as specified on the Competition Website.
b. Teams. If permitted under the Competition Website guidelines, multiple individuals may collaborate as a Team; however, you may join or form only one Team. Each Team member must be a single individual with a separate Kaggle account. You must register individually for the Competition before joining a Team. You must confirm your Team membership to make it official by responding to the Team notification message sent to your Kaggle account. Team membership may not exceed the Maximum Team Size stated on the Competition Website.
c. Team Merger. Teams (or individual Participants) may request to merge via the Competition Website. Team mergers may be allowed provided that: (i) the combined Team does not exceed the Maximum Team Size; (ii) the number of Submissions made by the merging Teams does not exceed the number of Submissions permissible for one Team at the date of the merger request; (iii) the merger is completed before the earlier of: any merger deadline or the Competition deadline; and (iv) the proposed combined Team otherwise meets all the requirements of these Rules.
d. Private Sharing. No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums.

6. SUBMISSION CODE REQUIREMENTS
a. Private Code Sharing. Unless otherwise specifically permitted under the Competition Website or Competition Specific Rules above, during the Competition Period, you are not allowed to privately share source or executable code developed in connection with or based upon the Competition Data or other source or executable code relevant to the Competition (“Competition Code”). This prohibition includes sharing Competition Code between separate Teams, unless a Team merger occurs. Any such sharing of Competition Code is a breach of these Competition Rules and may result in disqualification.
b. Public Code Sharing. You are permitted to publicly share Competition Code, provided that such public sharing does not violate the intellectual property rights of any third party. If you do choose to share Competition Code or other such code, you are required to share it on Kaggle.com on the discussion forum or notebooks associated specifically with the Competition for the benefit of all competitors. By so sharing, you are deemed to have licensed the shared code under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such Competition Code or model containing or depending on such Competition Code.
c. Use of Open Source. Unless otherwise stated in the Specific Competition Rules above, if open source code is used in the model to generate the Submission, then you must only use open source code licensed under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such code or model containing or depending on such code.

7. DETERMINING WINNERS
a. Each Submission will be scored and/or ranked by the evaluation metric, or Evaluation Rubric (in the case of Hackathon Competitions),stated on the Competition Website. During the Competition Period, the current ranking will be visible on the Competition Website's Public Leaderboard. The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard, subject to compliance with these Rules. The Public Leaderboard will be based on the public test set and the Private Leaderboard will be based on the private test set. There will be no leaderboards for Hackathon Competitions.
b. In the event of a tie, the Submission that was entered first to the Competition will be the winner. In the event a potential winner is disqualified for any reason, the Submission that received the next highest score rank will be chosen as the potential winner. For Hackathon Competitions, each of the top Submissions will get a unique ranking and there will be no tiebreakers.

8. NOTIFICATION OF WINNERS & DISQUALIFICATION
a. The potential winner(s) will be notified by email.
b. If a potential winner (i) does not respond to the notification attempt within one (1) week from the first notification attempt or (ii) notifies Kaggle within one week after the Final Submission Deadline that the potential winner does not want to be nominated as a winner or does not want to receive a Prize, then, in each case (i) and (ii) such potential winner will not receive any Prize, and an alternate potential winner will be selected from among all eligible entries received based on the Competition’s judging criteria.
c. In case (i) and (ii) above Kaggle may disqualify the Participant. However, in case (ii) above, if requested by Kaggle, such potential winner may provide code and documentation to verify the Participant’s compliance with these Rules. If the potential winner provides code and documentation to the satisfaction of Kaggle, the Participant will not be disqualified pursuant to this paragraph.
d. Competition Sponsor reserves the right to disqualify any Participant from the Competition if the Competition Sponsor reasonably believes that the Participant has attempted to undermine the legitimate operation of the Competition by cheating, deception, or other unfair playing practices or abuses, threatens or harasses any other Participants, Competition Sponsor or Kaggle.
e. A disqualified Participant may be removed from the Competition leaderboard, at Kaggle's sole discretion. If a Participant is removed from the Competition Leaderboard, additional winning features associated with the Kaggle competition platform, for example Kaggle points or medals, may also not be awarded.
f. The final leaderboard list will be publicly displayed at Kaggle.com. Determinations of Competition Sponsor are final and binding.

9. PRIZES
a. Prize(s) are as described on the Competition Website and are only available for winning during the time period described on the Competition Website. The odds of winning any Prize depends on the number of eligible Submissions received during the Competition Period and the skill of the Participants.
b. All Prizes are subject to Competition Sponsor's review and verification of the Participant’s eligibility and compliance with these Rules, and the compliance of the winning Submissions with the Submissions Requirements. In the event that the Submission demonstrates non-compliance with these Competition Rules, Competition Sponsor may at its discretion take either of the following actions: (i) disqualify the Submission(s); or (ii) require the potential winner to remediate within one week after notice all issues identified in the Submission(s) (including, without limitation, the resolution of license conflicts, the fulfillment of all obligations required by software licenses, and the removal of any software that violates the software restrictions).
c. A potential winner may decline to be nominated as a Competition winner in accordance with Section 3.8.
d. Potential winners must return all required Prize acceptance documents within two (2) weeks following notification of such required documents, or such potential winner will be deemed to have forfeited the prize and another potential winner will be selected. Prize(s) will be awarded within approximately thirty (30) days after receipt by Competition Sponsor or Kaggle of the required Prize acceptance documents. Transfer or assignment of a Prize is not allowed.
e. You are not eligible to receive any Prize if you do not meet the Eligibility requirements in Section 2.7 and Section 3.1 above.
f. If a Team wins a monetary Prize, the Prize money will be allocated in even shares between the eligible Team members, unless the Team unanimously opts for a different Prize split and notifies Kaggle before Prizes are issued.

10. TAXES
a. ALL TAXES IMPOSED ON PRIZES ARE THE SOLE RESPONSIBILITY OF THE WINNERS. Payments to potential winners are subject to the express requirement that they submit all documentation requested by Competition Sponsor or Kaggle for compliance with applicable state, federal, local and foreign (including provincial) tax reporting and withholding requirements. Prizes will be net of any taxes that Competition Sponsor is required by law to withhold. If a potential winner fails to provide any required documentation or comply with applicable laws, the Prize may be forfeited and Competition Sponsor may select an alternative potential winner. Any winners who are U.S. residents will receive an IRS Form-1099 in the amount of their Prize.

11. GENERAL CONDITIONS
a. All federal, state, provincial and local laws and regulations apply.

12. PUBLICITY
a. You agree that Competition Sponsor, Kaggle and its affiliates may use your name and likeness for advertising and promotional purposes without additional compensation, unless prohibited by law.

13. PRIVACY
a. You acknowledge and agree that Competition Sponsor and Kaggle may collect, store, share and otherwise use personally identifiable information provided by you during the Kaggle account registration process and the Competition, including but not limited to, name, mailing address, phone number, and email address (“Personal Information”). Kaggle acts as an independent controller with regard to its collection, storage, sharing, and other use of this Personal Information, and will use this Personal Information in accordance with its Privacy Policy <www.kaggle.com/privacy>, including for administering the Competition. As a Kaggle.com account holder, you have the right to request access to, review, rectification, portability or deletion of any personal data held by Kaggle about you by logging into your account and/or contacting Kaggle Support at <www.kaggle.com/contact>.
b. As part of Competition Sponsor performing this contract between you and the Competition Sponsor, Kaggle will transfer your Personal Information to Competition Sponsor, which acts as an independent controller with regard to this Personal Information. As a controller of such Personal Information, Competition Sponsor agrees to comply with all U.S. and foreign data protection obligations with regard to your Personal Information. Kaggle will transfer your Personal Information to Competition Sponsor in the country specified in the Competition Sponsor Address listed above, which may be a country outside the country of your residence. Such country may not have privacy laws and regulations similar to those of the country of your residence.

14. WARRANTY, INDEMNITY AND RELEASE
a. You warrant that your Submission is your own original work and, as such, you are the sole and exclusive owner and rights holder of the Submission, and you have the right to make the Submission and grant all required licenses. You agree not to make any Submission that: (i) infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; or (ii) otherwise violates any applicable U.S. or foreign state or federal law.
b. To the maximum extent permitted by law, you indemnify and agree to keep indemnified Competition Entities at all times from and against any liability, claims, demands, losses, damages, costs and expenses resulting from any of your acts, defaults or omissions and/or a breach of any warranty set forth herein. To the maximum extent permitted by law, you agree to defend, indemnify and hold harmless the Competition Entities from and against any and all claims, actions, suits or proceedings, as well as any and all losses, liabilities, damages, costs and expenses (including reasonable attorneys fees) arising out of or accruing from: (a) your Submission or other material uploaded or otherwise provided by you that infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; (b) any misrepresentation made by you in connection with the Competition; (c) any non-compliance by you with these Rules or any applicable U.S. or foreign state or federal law; (d) claims brought by persons or entities other than the parties to these Rules arising from or related to your involvement with the Competition; and (e) your acceptance, possession, misuse or use of any Prize, or your participation in the Competition and any Competition-related activity.
c. You hereby release Competition Entities from any liability associated with: (a) any malfunction or other problem with the Competition Website; (b) any error in the collection, processing, or retention of any Submission; or (c) any typographical or other error in the printing, offering or announcement of any Prize or winners.

15. INTERNET
a. Competition Entities are not responsible for any malfunction of the Competition Website or any late, lost, damaged, misdirected, incomplete, illegible, undeliverable, or destroyed Submissions or entry materials due to system errors, failed, incomplete or garbled computer or other telecommunication transmission malfunctions, hardware or software failures of any kind, lost or unavailable network connections, typographical or system/human errors and failures, technical malfunction(s) of any telephone network or lines, cable connections, satellite transmissions, servers or providers, or computer equipment, traffic congestion on the Internet or at the Competition Website, or any combination thereof, which may limit a Participant’s ability to participate.

16. RIGHT TO CANCEL, MODIFY OR DISQUALIFY
a. If for any reason the Competition is not capable of running as planned, including infection by computer virus, bugs, tampering, unauthorized intervention, fraud, technical failures, or any other causes which corrupt or affect the administration, security, fairness, integrity, or proper conduct of the Competition, Competition Sponsor reserves the right to cancel, terminate, modify or suspend the Competition. Competition Sponsor further reserves the right to disqualify any Participant who tampers with the submission process or any other part of the Competition or Competition Website. Any attempt by a Participant to deliberately damage any website, including the Competition Website, or undermine the legitimate operation of the Competition is a violation of criminal and civil laws. Should such an attempt be made, Competition Sponsor and Kaggle each reserves the right to seek damages from any such Participant to the fullest extent of the applicable law.

17. NOT AN OFFER OR CONTRACT OF EMPLOYMENT
a. Under no circumstances will the entry of a Submission, the awarding of a Prize, or anything in these Rules be construed as an offer or contract of employment with Competition Sponsor or any of the Competition Entities. You acknowledge that you have submitted your Submission voluntarily and not in confidence or in trust. You acknowledge that no confidential, fiduciary, agency, employment or other similar relationship is created between you and Competition Sponsor or any of the Competition Entities by your acceptance of these Rules or your entry of your Submission.

18. DEFINITIONS
a. "Competition Data" are the data or datasets available from the Competition Website for the purpose of use in the Competition, including any prototype or executable code provided on the Competition Website. The Competition Data will contain private and public test sets. Which data belongs to which set will not be made available to Participants.
b. An “Entry” is when a Participant has joined, signed up, or accepted the rules of a competition. Entry is required to make a Submission to a competition.
c. A “Final Submission” is the Submission selected by the user, or automatically selected by Kaggle in the event not selected by the user, that is/are used for final placement on the competition leaderboard.
d. A “Participant” or “Participant User” is an individual who participates in a competition by entering the competition and making a Submission.
e. The “Private Leaderboard” is a ranked display of Participants’ Submission scores against the private test set. The Private Leaderboard determines the final standing in the competition.
f. The “Public Leaderboard” is a ranked display of Participants’ Submission scores against a representative sample of the test data. This leaderboard is visible throughout the competition.
g. A “Sponsor” is responsible for hosting the competition, which includes but is not limited to providing the data for the competition, determining winners, and enforcing competition rules.
h. A “Submission” is anything provided by the Participant to the Sponsor to be evaluated for competition purposes and determine leaderboard position. A Submission may be made as a model, notebook, prediction file, or other format as determined by the Sponsor.
i. A “Team” is one or more Participants participating together in a Kaggle competition, by officially merging together as a Team within the competition platform.