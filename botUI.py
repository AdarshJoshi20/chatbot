import json
import numpy as np
import re
from difflib import get_close_matches
from keras.models import load_model, Model
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from tkinter import Tk, Frame, Scrollbar, Canvas, Label, Entry, Button, END, VERTICAL, Y

class ChatBot:
    def __init__(self, root):
        # Load the seq2seq model
        self.training_model = load_model('training_model.h5')
        encoder_inputs = self.training_model.input[0]
        encoder_outputs, state_h_enc, state_c_enc = self.training_model.layers[2].output
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = Model(encoder_inputs, encoder_states)

        latent_dim = 256
        decoder_state_input_hidden = Input(shape=(latent_dim,))
        decoder_state_input_cell = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

        # Decoder
        dimensionality = 256
        num_decoder_tokens = 3101
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_states = [decoder_state_input_hidden, decoder_state_input_cell]
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        # Load the knowledge base
        self.knowledge_base = self.load_knowledge_base('knowledge_base.json')
        self.question_dict = self.build_question_dict(self.knowledge_base["questions"])

        self.negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
        self.exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop")

        self.root = root
        self.create_ui()

    def load_knowledge_base(self, file_path: str) -> dict:
        with open(file_path, 'r', encoding='utf-8') as file:
            data: dict = json.load(file)
        return data

    def build_question_dict(self, questions: list) -> dict:
        return {q["Question"]: q["Answer"] for q in questions}

    def save_knowledge_base(self, file_path: str, data: dict):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)

    def find_best_match(self, user_question: str) -> str | None:
        matches: list = get_close_matches(user_question, self.question_dict.keys(), n=1, cutoff=0.6)
        return matches[0] if matches else None

    def get_answer_for_question(self, question: str) -> str | None:
        return self.question_dict.get(question)

    def string_to_matrix(self, user_input):
        num_encoder_tokens = 282
        max_encoder_seq_length = 53
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
        user_input_matrix = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens),
            dtype='float32'
        )
        with open('input_features_dict', 'r') as json_file:
            input_features_dict = json.load(json_file)
        for timestep, token in enumerate(tokens):
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.
        return user_input_matrix

    def decode_response(self, test_input):
        num_decoder_tokens = 3101
        max_decoder_seq_length = 1642
        with open('target_features_dict', 'r') as json_file:
            target_features_dict = json.load(json_file)
        states_value = self.encoder_model.predict(test_input)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, target_features_dict['<START>']] = 1.
        with open('reverse_target_features_dict', 'r') as json_file:
            reverse_target_features_dict = json.load(json_file)
        decoded_sentence = ''
        stop_condition = False

        while not stop_condition:
            output_tokens, hidden_state, cell_state = self.decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_features_dict[sampled_token_index]
            decoded_sentence += " " + sampled_token

            if sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            states_value = [hidden_state, cell_state]

        return decoded_sentence

    def start_chat(self):
        self.user_input.delete(0, END)
        self.update_chat_history("Hi, I'm a chatbot trained on random dialogs. AMA!\n")
        self.root.mainloop()

    def chat(self):
        user_input = self.user_input.get()
        if user_input:
            self.update_chat_history(f"You: {user_input}\n")

            # Process the user's input and get the chatbot's response
            chatbot_response = self.generate_response(user_input)
            self.update_chat_history(f"ChatBot: {chatbot_response}\n")

            self.user_input.delete(0, END)

    def generate_response(self, user_input):
        input_matrix = self.string_to_matrix(user_input)
        chatbot_response = self.decode_response(input_matrix)
        chatbot_response = chatbot_response.replace("<START>", '')
        chatbot_response = chatbot_response.replace("<END>", '')
        return chatbot_response

    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply:
                self.update_chat_history("Ok, have a great day!\n")
                self.root.destroy()

    def update_chat_history(self, message):
        self.chat_history += message
        self.output.create_text(10, 10, anchor="w", text=self.chat_history, font=("Arial", 12), fill="black")
        self.output.config(scrollregion=self.output.bbox("all"))
        self.output.yview(END)

    def create_ui(self):
        self.root.title("ChatBot")
        self.root.geometry("400x500")

        self.output_frame = Frame(self.root)
        self.output_frame.pack(pady=10)

        self.scrollbar = Scrollbar(self.output_frame, orient=VERTICAL)
        self.output = Canvas(self.output_frame, yscrollcommand=self.scrollbar.set)
        self.output.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.scrollbar.config(command=self.output.yview)

        self.user_input = Entry(self.root, width=40)
        self.user_input.pack(pady=10)

        self.send_button = Button(self.root, text="Send", command=self.chat)
        self.send_button.pack()

        self.chat_history = ""

# Run the chatbot UI
if __name__ == '__main__':
    root = Tk()
    chatbot = ChatBot(root)
    chatbot.start_chat()
