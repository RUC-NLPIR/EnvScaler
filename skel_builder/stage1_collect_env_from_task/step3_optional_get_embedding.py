"""
step3-1 (optional): Generate embeddings for environment descriptions for deduplication.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))   

from tqdm import tqdm

from utils.call_llm import openai_batch_embedding_inference
from utils.process_file import read_file, save_file


def add_embeddings_to_samples(samples, field, model, timeout):
    """Generate embeddings for a batch of samples and add them to each sample."""
    # Extract all input texts
    inputs = [sample[field] for sample in samples]

    # Generate embeddings via API
    embeddings = openai_batch_embedding_inference(
        model=model,
        texts=inputs,
    )
    for sample, embedding in zip(samples, embeddings):
        sample[field + "_embedding"] = embedding
    return samples



def batch_add_embeddings(data, field, model, batch_size, timeout=60):
    """Process data in batches and add embeddings to each batch."""
    new_data = []
    # Process in batches
    for i in tqdm(range(0, len(data), batch_size), desc="Embedding batches"):
        batch = data[i:i + batch_size]
        batch_with_emb = add_embeddings_to_samples(samples=batch, field=field, model=model, timeout=timeout)
        new_data.extend(batch_with_emb)
    return new_data


if __name__ == "__main__":
    samples = read_file('stage1_collect_env_from_task/temp_result/step2_infered_env_description.json')
    for sample in samples:
        sample['env_summary_and_introduction'] = f"**{sample['environment_summary']}**: {sample['environment_introduction']}"
    field = "env_summary_and_introduction"
    model = "text-embedding-3-large"
    batch_size = 2
    samples = batch_add_embeddings(data=samples, field=field, model=model, batch_size=batch_size)
    save_file('stage1_collect_env_from_task/temp_result/step3_infered_env_description_with_embedding.json', samples)