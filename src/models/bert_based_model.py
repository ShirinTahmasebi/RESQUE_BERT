from transformers import PreTrainedModel, BertPreTrainedModel, RobertaPreTrainedModel, BertModel, RobertaModel
from imports import *
import sys
sys.path.append('../')


class ResqueModel(PreTrainedModel, ABC):

    def __init__(self, config):
        super().__init__(config)

        # Initialize the configurations of the bert model
        config.num_labels = 2
        config.output_hidden_states = True
        config.return_dict = True
        self.num_labels = config.num_labels

        # Initialize the bert model
        self.bert_model = self.initialize_model(config)

        # Initialize the classifier
        self.classifier = nn.Linear(
            self.bert_model.config.hidden_size,
            self.num_labels
        )

        self.init_weights()

    @abstractmethod
    def initialize_model(self, config) -> PreTrainedModel:
        raise NotImplementedError(
            'The model should be initialized in the children of this class!'
        )

    @abstractmethod
    def maximum_input_length(self) -> int:
        raise NotImplementedError(
            'The maximum number of tokens in the input differs from model to model. \
                So, it should be specified in the children of this class!'
        )

    @abstractmethod
    def get_bert_inputs(self, **args) -> dict:
        raise NotImplementedError(
            'Since the input varies for different BERT variants, specify the inputs in the children classes.'
        )

    def forward(self, input_ids, atttention_mask, token_type_ids, cls_mask, labels):
        for i in range(len(input_ids)):
            assert len(input_ids[i]) == self.maximum_input_length()
            assert len(atttention_mask[i]) == self.maximum_input_length()
            assert len(token_type_ids[i]) == self.maximum_input_length()
            assert len(cls_mask[i]) == self.maximum_input_length()
            assert len(labels[i]) == self.maximum_input_length()

        labels = labels[labels != -1].to(torch.long)

        filtered_bert_inputs = self.get_bert_inputs(
            input_ids=input_ids,
            attention_mask=atttention_mask,
            token_type_ids=token_type_ids
        )

        bert_output = self.bert_model(**filtered_bert_inputs, return_dict=True)

        indices_of_cls_tokens_as_tuple = (cls_mask == 1).nonzero(as_tuple=True)

        list_of_cls_hidden_states = []
        for i in range(len(bert_output.last_hidden_state)):
            temp = indices_of_cls_tokens_as_tuple[1][indices_of_cls_tokens_as_tuple[0] == i]
            list_of_cls_hidden_states.append(
                bert_output.last_hidden_state[i][temp]
            )

        tensor_of_cls_hidden_states = torch.cat(list_of_cls_hidden_states)
        classifier_output = self.classifier(tensor_of_cls_hidden_states)
        predicted_labels = torch.argmax(classifier_output, 1)

        loss_ce = nn.CrossEntropyLoss()
        loss = loss_ce(
            classifier_output.view(-1, self.num_labels),
            labels.view(-1)
        )

        return loss, predicted_labels


class ResqueRoBertaModel(ResqueModel, RobertaPreTrainedModel):

    def initialize_model(self, config):
        return RobertaModel(config)

    def maximum_input_length(self):
        return 512

    def get_bert_inputs(self, **args):
        assert args.__contains__('input_ids')
        assert args.__contains__('attention_mask')

        return {
            'input_ids': args['input_ids'],
            'attention_mask': args['attention_mask']
        }


class ResqueBertModel(ResqueModel, BertPreTrainedModel):

    def initialize_model(self, config):
        return BertModel(config)

    def maximum_input_length(self):
        return 512

    def get_bert_inputs(self, **args):
        assert args.__contains__('input_ids')
        assert args.__contains__('attention_mask')
        assert args.__contains__('token_type_ids')

        return {
            'input_ids': args['input_ids'],
            'attention_mask': args['attention_mask'],
            'token_type_ids': args['token_type_ids'],
        }
