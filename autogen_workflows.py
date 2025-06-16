import json
from typing import Any, Dict, List
from jinja2 import Template, Environment, FileSystemLoader

from os import makedirs
from os.path import exists as file_is_exists



class AutogenWorkflows:
  CONTRACTS_PATH:      str = './contracts'
  GITHUB_ACTIONS_PATH: str = './.github/workflows'

  environment: Environment = Environment(
    loader = FileSystemLoader(CONTRACTS_PATH)
  )

  def generate_by_model_registries(
    self, model_registries: List[str]
  ) -> None:
    try:
      if not file_is_exists(self.GITHUB_ACTIONS_PATH):
        makedirs(self.GITHUB_ACTIONS_PATH)

      for _idx, registry in enumerate(model_registries):
        template: Template = self.environment \
          .get_template('workflow.yaml.jinja2')

        template_context: Dict[str, Any] = {
          'workflow_name':  f'Automated Training [Workflow - {_idx + 1}]',

          'registry_name':  registry.get('registry_name'),
          'registry_url' :  registry.get('registry_url'),

          'workloads_file':  f'workloads_{_idx + 1}.json',

          'python_version': '3.11',
          'script_name':    'main.py'
        }

        template_render: str = template.render(template_context)
        with open(f'{self.GITHUB_ACTIONS_PATH}/workflow_{_idx + 1}_pipeline.yaml', 'w') \
          as workflow_file: workflow_file.write(template_render)
        
      print('generate workflow is success..')

    except Exception as error_message:
      print(error_message)


if __name__ == '__main__':
  autogen: AutogenWorkflows = AutogenWorkflows()
  
  with open('train_config.json', 'r') as json_file:
    registries: List[str] = json.load(json_file) \
      .get('model_registries', [])

  autogen.generate_by_model_registries(
    model_registries = registries
  )
