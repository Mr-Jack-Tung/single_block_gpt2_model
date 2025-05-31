# Single Block GPT-2 (No Dependencies)

This project provides a minimal implementation of a GPT-2-like language model with a single transformer block and a character-level tokenizer, built from scratch using PyTorch without relying on external libraries like Hugging Face `transformers`. It is primarily intended for educational or experimental purposes to understand the basic components of a transformer model.

Ở bản thử nghiệm này, tôi cố gắng không sử dụng thư viện transformers của huggingface vì nó có thể gây khó khăn khi hiểu rõ hơn về cơ chế hoạt động của mô hình transformer. Tuy nhiên, nếu bạn muốn sử dụng thư viện transformers thì bạn có thể tham khảo các phiên bản khác của dự án nhé.

Chúng tôi đã phát hiện ra rằng việc huấn luyện mô hình trong các giai đoạn riêng biệt với khởi tạo lại của mô hình, bộ tối ưu hóa và mã hóa ký tự ở đầu mỗi giai đoạn, sau đó chỉ tải trạng thái của mô hình được lưu trữ là một cách hiệu quả để đạt được sự trùng khớp chính xác của dữ liệu huấn luyện này. Mô hình được cài đặt cho hai giai đoạn với 30 epoch mỗi giai đoạn, điều này đã được chứng minh là cấu hình thành công.

Mình phát hiện ra một điểm rất thú vị là chạy quá trình training model 2 lần với 30 epoch mỗi lần thì cho ra kết quả chính xác, còn nếu chạy training 1 lần với 300 epoch, thậm chí là 3000 epoch thì kết quả output vẫn không chính xác ?! ... đố bác nào biết lý do vì sao có hiện tượng này nhé ^^

## Project Structure

- `single_block_gpt2_no_depend_model.py`: Defines the model architecture, including the `GPT2Config`, `SelfAttention`, `MLP`, `TransformerBlock`, and the main `SingleBlockGPT2ModelNoDepend` class.
- `character_tokenizer.py`: Implements a simple character-level tokenizer.
- `train_single_block_gpt2_no_depend.py`: Script for training the model.
- `inference_single_block_gpt2_no_depend.py`: Script for generating text using a trained model.

## Getting Started

1.  Ensure you have Python and PyTorch installed. You can install PyTorch by following the instructions on the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2.  Navigate to the `Single_Block_GPT2_No_depend` directory in your terminal.

## Usage

### Training the Model

The `train_single_block_gpt2_no_depend.py` script is configured to train the model on a single QA pair: 'Question: Xin chào \nAnswer: Công ty BICweb kính chào quý khách!.'.

Based on our research, training the model in stages with re-initialization of the model, optimizer, and tokenizer at the start of each stage, and loading only the model state, is effective for achieving accurate reproduction of this specific training data. The script is currently set up for 2 stages of 30 epochs each, which was found to be a successful configuration.

To train the model, run the following command from the project root directory:

```bash
uv run Single_Block_GPT2_No_depend/train_single_block_gpt2_no_depend.py
```

This will perform the staged training and save the trained model state and tokenizer in the `TrainedSingleBlockGPT2_No_depend` directory.

### Generating Text (Inference)

The `inference_single_block_gpt2_no_depend.py` script uses the trained model to generate text. It is configured to use greedy decoding and generate a fixed number of tokens (64) to demonstrate the model's ability to reproduce the training data.

To run inference using the trained model, run the following command from the project root directory:

```bash
uv run Single_Block_GPT2_No_depend/inference_single_block_gpt2_no_depend.py
```

This will load the trained model and tokenizer and generate text based on the input prompt "Question: Xin chào".

## Key Findings

During the research and experimentation with this project, we made a significant observation regarding the training process and its impact on the model's ability to accurately reproduce the single training data point:

-   **Staged Training Effectiveness:** Running the training process in multiple distinct stages (e.g., 2 stages of 30 epochs each), saving and loading the model state between stages, was found to be crucial for achieving accurate reproduction of the training data during inference.
-   **Optimizer Re-initialization:** A key factor contributing to the effectiveness of staged training appears to be the **re-initialization of the optimizer** at the start of each training stage. Unlike a single continuous training run or staged training that preserves the optimizer's internal state, resetting the optimizer periodically seems to help the optimization process navigate the complex loss landscape when overfitting this specific, minimal dataset, leading to a better final model state for perfect character-level prediction.
-   **Tokenizer Interaction:** While the custom model architecture itself is functional (as shown by successful inference when paired with a different tokenizer), achieving accurate reproduction with the custom character tokenizer required careful tuning of the training process, highlighting the interaction between the tokenizer's representation and the model's ability to overfit.

This project serves as an interesting case study on the nuances of training dynamics, particularly in extreme overfitting scenarios with minimal data and simple model components.
