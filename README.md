# Interpreting Social Contingency in a Conversation
## Maciej Zielonka
### Written for Yale CPSC 490 Final Project
### Mentored by Rebecca Ramnauth

There were two goals for this project: 

1. To create a classifier that can create binary predictions regarding engagement based on low-level audio features of speech
2. To investigate which structural features of a conversation contribute to our understanding of Social Contingency

**Social Contingency**, as defined for this project, is one's sensitivity and responsiveness to another social agent's speech and behavior.

The folders for this project are:
* **Interview_data** - Debrief interviews obtained from the data from a [study](https://dl.acm.org/doi/10.1145/2696454.2696466) run by Iolanda Leite and the Social Robotics Lab at Yale
* **ELAN_Annotations** - .eaf files created using ELAN software, marking down sections of engagement and disengagement 
* **Annotations_as_json** - .json files storing the above annotations as dictionaries
* **src** - All the relevant code, including helper functions, and the notebooks that perform model evaluation, feature extraction, and diarization
* **structural_features_csvs** - .csv files containing structural features of conversation
* **wav_chunk_csvs** - .csv files containing segments of wav files, which correspond to the annotated sections of the interview data 