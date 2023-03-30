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
        if type == InputSequenceTypeEnum.PROMPT_V2:
            return TemplateCreationPromptV2()

    @abstractmethod
    def create_prompt_from_single_row(self, input_row):
        raise NotImplementedError(
            'The prompt creation should be implemented!'
        )

    @abstractmethod
    def separator_text(self):
        raise NotImplementedError('The separator should be defined!')

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
        return '[SEP]'


class TemplateCreationPromptV2(TemplateCreation):
    def create_prompt_from_single_row(self, input_row):
        pass

    def separator_text(self):
        return '[SEP]'


def execute_input_creation(input_data_path, output_data_path, type: InputSequenceTypeEnum):
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
    output_df.to_csv(output_data_path)
    pass


def execute_tokenization(input_data, output_data):
    input_df = pd.read_csv(get_absolute_path(input_data))

    pass
