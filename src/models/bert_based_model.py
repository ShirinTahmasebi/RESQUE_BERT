import sys
sys.path.append('../')

from imports import *

class ResqueModel(nn.Module):
    def __init__(self, bert_type, number_of_classes):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(bert_type, output_hidden_states = True, return_dict=True)
        self.classifier = nn.Linear(self.bert_model.config.hidden_size, number_of_classes)
        self.criterion = nn.BCELoss()
    
    def forward(self, input_ids, atttention_mask, token_type_ids, cls_mask, labels):
        labels = labels[labels != -1].to(torch.float)

        bert_output = self.bert_model(
            input_ids, 
            attention_mask=atttention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        indices_of_cls_tokens_as_tuple = (cls_mask == 1).nonzero(as_tuple=True)

        list_of_cls_hidden_states = []
        for i in range(len(bert_output.last_hidden_state)):
            temp = indices_of_cls_tokens_as_tuple[1][indices_of_cls_tokens_as_tuple[0] == i]
            list_of_cls_hidden_states.append(bert_output.last_hidden_state[i][temp])

        tensor_of_cls_hidden_states = torch.cat(list_of_cls_hidden_states)
        classifier_output = self.classifier(tensor_of_cls_hidden_states)
        final_output = torch.sigmoid(classifier_output)

        # Calculate the loss value
        predicted_labels = torch.argmax(final_output, 1) 
        # Argmax is not differentiable. So, we need to manually make the output differentiable.
        predicted_labels = predicted_labels.to(torch.float).requires_grad_()
        loss = self.criterion(predicted_labels, labels)

        return loss, predicted_labels


