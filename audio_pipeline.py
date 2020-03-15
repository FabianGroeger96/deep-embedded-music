from input_pipeline.triplet_input_pipeline import TripletsInputPipeline

if __name__ == "__main__":
    pipeline = audio_pipeline = TripletsInputPipeline(
        audio_files_path="/opt/project/datasets/DCASE18-Task5-development/",
        info_file_path="/opt/project/datasets/DCASE18-Task5-development/evaluation_setup/fold1_train.txt",
        sample_rate=16000,
        sample_size=10,
        batch_size=64,
        prefetch_batches=128,
        input_processing_buffer_size=1)
    pipeline.generate_samples()
