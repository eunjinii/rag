from datasets import load_dataset, concatenate_datasets
import pandas as pd

class QuestionAnswering:
    def __init__(self, dataset_type):
        
        DATASET_DICT = {
            "MedMCQA": "openlifescienceai/medmcqa", # 4 choices / subject
            "MedQA": "GBaker/MedQA-USMLE-4-options-hf", # 4 choices / no subject
            "MMLU": "cais/mmlu", # 4 chioces
            "MedQuad": "keivalya/MedQuad-MedicalQnADataset", # short answer
            # "MedTrans": "tchebonenko/MedicalTranscriptions",
            # "MedKeywords": "argilla/medical-keywords"
        }

        assert dataset_type in DATASET_DICT, "Invalid dataset type"

        self.dataset_type = dataset_type
        self.dataset_name = DATASET_DICT[self.dataset_type]
        self.dataset = self.load_test_dataset()
        self.qa_df = None
    
    def load_test_dataset(self):
        print("Loading dataset...")
        if self.dataset_type in ["MedMCQA", "MedQuad"]:
            dataset = load_dataset(self.dataset_name, split="train")
            split_dataset = dataset.train_test_split(test_size=0.1)
            test_dataset = split_dataset["test"]
            return test_dataset

        elif self.dataset_type == "MMLU":
            subject_list = ["anatomy", "clinical_knowledge", "college_biology", "college_medicine", "human_aging", "medical_genetics", "nutrition", "professional_medicine"]
            all_datasets = []
            for subject in subject_list:
                dataset = load_dataset("NLPCoreTeam/mmlu_ru", name=subject, split="test")
                dataset = dataset.map(lambda example: {**example, "subject": subject})
                all_datasets.append(dataset)
            test_dataset = concatenate_datasets(all_datasets)
            return test_dataset

        elif self.dataset_type == "MedQA":
            test_dataset = load_dataset(self.dataset_name, split="test")
            return test_dataset

    def get_question_answering_dataframe(self):
        """
        Converts dataset to DataFrame and processes it into QnA format.
        question, options ["A", "B", "C", "A, B, and C"], answer(a, b, c, or d), subject(optional)
        """ 
        
        if not self.dataset:
            raise ValueError("Dataset not loaded properly.")
        
        df = self.dataset.to_pandas()
        ANSWER_MAPPING_DICT = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
        
        if self.dataset_type == "MedMCQA":
            df.rename(columns={"cop": "answerindex", "subject_name": "subject"}, inplace=True)
            df['answer'] = df['answerindex'].map(ANSWER_MAPPING_DICT)
            df['options'] = df.apply(lambda row: [row['opa'], row['opb'], row['opc'], row['opd']], axis=1)
            df.drop(columns=["id", "opa", "opb", "opc", "opd", "exp", "topic_name", "choice_type", "answerindex"], inplace=True)
            
        elif self.dataset_type == "MedQA":
            df.rename(columns={"sent1": "question"}, inplace=True)
            df['answer'] = df['label'].map(ANSWER_MAPPING_DICT)
            df['options'] = df.apply(lambda row: [row['ending0'], row['ending1'], row['ending2'], row['ending3']], axis=1)
            df.drop(columns=["id", "sent2", "ending0", "ending1", "ending2", "ending3", "label"], inplace=True)

        elif self.dataset_type == "MMLU":
            df.rename(columns={"question_en": "question", "choices_en":"options"}, inplace=True)
            df['answer'] = df['answer'].map(ANSWER_MAPPING_DICT)
            df.drop(columns=["question_ru", "choices_ru"], inplace=True)
        
        # elif self.dataset_type == "MedQuad":
        #     df.rename(columns={"Question": "question", "Answer":"answer"}, inplace=True)
        
        self.qa_df = df
        return self.qa_df
    
    def get_subfield_list(self):
        if self.dataset_type in ["MedMCQA", ""]:
            return self.question_answering_dataframe['subject'].unique()
        return None

# qa = QuestionAnswering("MMLU")
# qa.get_question_answering_dataframe()
# print(qa.qa_df.head())