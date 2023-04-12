from utils.template_constants import *
from utils.helper import get_absolute_path
from imports import *
import sys
sys.path.append('../')


class TemplateCreation(ABC):
    def instantiate(type: InputSequenceTypeEnum):
        if type == InputSequenceTypeEnum.SIMPLE:
            return TemplateCreationSimple()
        if type == InputSequenceTypeEnum.PROMPT_V1:
            return TemplateCreationPromptV1()

    @abstractmethod
    def create_prompt_from_single_row(self, input_row):
        raise NotImplementedError(
            'The prompt creation should be implemented!'
        )

    @abstractmethod
    def separator_text(self):
        raise NotImplementedError('The separator should be defined!')

    @abstractmethod
    def task_2_label_text(self):
        raise NotImplementedError('The separator should be defined!')

    @abstractmethod
    def extract_index_of_labels(self, prompted_input):
        raise NotImplementedError('The idex extractor should be defined!')

    def create_prompt(self, batch_input_data):
        output_rows = batch_input_data.apply(
            lambda x: self.create_prompt_from_single_row(x),
            axis=1
        )

        output_prompt = self.separator_text().join(output_rows.values)
        return output_prompt


class TemplateCreationSimple(TemplateCreation):
    def create_prompt_from_single_row(self, input_row):
        assert 'template_label' in input_row, 'Wrong input is passed to template creation class!'
        return input_row['template_label']

    def separator_text(self):
        return ' '

    def task_2_label_text(self):
        return ''

    def extract_index_of_labels(self, prompted_input):
        template_parts = prompted_input.split(self.separator_text())
        template_label_indices = list(range(len(template_parts)))
        return template_label_indices


class TemplateCreationPromptV1(TemplateCreation):
    def create_prompt_from_single_row(self, input_row):
        assert 'template_label' in input_row, 'Wrong input is passed to template creation class!'
        assert 'num_of_attributes_select' in input_row, 'Wrong input is passed to template creation class!'
        assert 'num_of_attributes_where' in input_row, 'Wrong input is passed to template creation class!'
        assert 'num_of_attributes_order_by' in input_row, 'Wrong input is passed to template creation class!'
        assert 'num_of_tables' in input_row, 'Wrong input is passed to template creation class!'

        return input_row['template_label'] + " " \
            + 'SELECT: ' + str(input_row['num_of_attributes_select']) + " " \
            + 'WHERE: ' + str(input_row['num_of_attributes_where']) + " " \
            + 'ORDER: ' + str(input_row['num_of_attributes_order_by']) + " " \
            + 'TABLES: ' + str(input_row['num_of_tables'])

    def separator_text(self):
        return ' [SEP] '

    def task_2_label_text(self):
        return 'The next template is:'

    def extract_index_of_labels(self, prompted_input):
        words = prompted_input.split()
        indices = [0]
        for i in range(len(words)):
            if words[i] == self.separator_text().strip():
                indices.append(i+1)
        return indices


def create_masked_input_for_task_1(input_sequence, instances_per_input: int, template_type: InputSequenceTypeEnum):
    template_creator = TemplateCreation.instantiate(type=template_type)
    template_label_indices = template_creator.extract_index_of_labels(
        input_sequence
    )

    import math
    number_of_masks = math.ceil(
        len(template_label_indices) * CONSTANTS.MASK_TOKEN_PERCENTAGE
    )

    set_of_mask_indices = set()
    number_of_attempts_max = instances_per_input + 20
    attempt_counter = 0
    while len(set_of_mask_indices) < instances_per_input and attempt_counter < number_of_attempts_max:
        mask_indices_arr = sorted(
            np.random.choice(
                template_label_indices,
                size=number_of_masks,
                replace=False
            )
        )
        set_of_mask_indices.add(tuple(mask_indices_arr))
        attempt_counter += 1

    input_texts = []
    labels = []

    for i in set_of_mask_indices:
        instance_input_splitted = input_sequence.split()
        instance_labels = []

        for j in list(i):
            instance_labels.append(instance_input_splitted[j])
            instance_input_splitted[j] = "[MASK]"

        input_texts.append(" ".join(instance_input_splitted))
        labels.append(instance_labels)

    return pd.DataFrame({'text': input_texts, 'labels': labels})


def create_masked_input_for_task_2(input_sequence, template_type: InputSequenceTypeEnum):
    template_creator = TemplateCreation.instantiate(type=template_type)
    template_label_indices = template_creator.extract_index_of_labels(
        input_sequence
    )

    MIN_SEQUENCE_LEN = 3

    if len(template_label_indices) < MIN_SEQUENCE_LEN:
        return pd.DataFrame({'text': [], 'labels': []})

    input_texts = []
    labels = []

    instance_input_splitted = input_sequence.split()
    for index in list(range(MIN_SEQUENCE_LEN - 1, len(template_label_indices) - 1)):
        end_index_of_input_exclusive = template_label_indices[index + 1]

        instance_input_list = instance_input_splitted[:end_index_of_input_exclusive]
        instance_input = " ".join(instance_input_list)
        instance_input += template_creator.separator_text()
        instance_input += f'{template_creator.task_2_label_text()} [MASK]'
        instance_input.strip()

        instance_label_index = template_label_indices[index + 1]
        instance_labels = [instance_input_splitted[instance_label_index]]

        input_texts.append(instance_input)
        labels.append(instance_labels)

    return pd.DataFrame({'text': input_texts, 'labels': labels})


def execute_input_creation(input_data_path: str, output_data_path: str, type: InputSequenceTypeEnum):
    input_df = pd.read_csv(get_absolute_path(input_data_path))
    unique_session_id_list = input_df['session_id'].unique()

    template_creator = TemplateCreation.instantiate(type=type)

    output_df_list = []
    for index, session_id in enumerate(unique_session_id_list):
        single_session_df = input_df[
            input_df['session_id'] == session_id
        ].copy()

        output_sequence = template_creator.create_prompt(single_session_df)
        output_df_list.append(output_sequence.strip())

        if index % 1000 == 0:
            print(f"Input creation is in progress: {index}")

    output_df = pd.DataFrame(output_df_list)
    output_df.to_csv(get_absolute_path(output_data_path))


def execute_masking_task_1(input_data_path: str, output_data_path: str, type: InputSequenceTypeEnum):
    input_df = pd.read_csv(get_absolute_path(input_data_path))

    output_rows = input_df.apply(
        lambda x: create_masked_input_for_task_1(
            input_sequence=x[1],
            instances_per_input=5,
            template_type=type
        ),
        axis=1
    )

    output_df = pd.concat(output_rows.to_numpy(), ignore_index=True)
    output_df.to_csv(get_absolute_path(output_data_path))


def execute_masking_task_2(input_data_path: str, output_data_path: str, type: InputSequenceTypeEnum):
    input_df = pd.read_csv(get_absolute_path(input_data_path))

    output_rows = input_df.apply(
        lambda x: create_masked_input_for_task_2(
            input_sequence=x[1],
            template_type=type
        ),
        axis=1
    )

    output_df = pd.concat(output_rows.to_numpy(), ignore_index=True)
    output_df.to_csv(get_absolute_path(output_data_path))
