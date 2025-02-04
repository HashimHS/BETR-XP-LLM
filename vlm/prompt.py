import os
import time
import openai
import json
import datetime
import numpy as np
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class VLMPrompter:
    def __init__(self, gpt_version="gpt-4o-mini", api_key=None, root_folder_path=None, task_name=None, skill_descriptions=None, plan_execution=None, scene_graph="scene_graph.txt", hierarchical_summary="hierarchical_summary.txt", images=None, failure_skill=None, failure_reason=None, resources=None) -> None:
        self.gpt_version = gpt_version
        if not api_key:
            raise ValueError("OpenAI API key is not provided.")
        openai.api_key = api_key
        self.root_folder_path = root_folder_path

        # Task-specific directory
        self.task_dir = os.path.join(root_folder_path, task_name)
        os.makedirs(self.task_dir, exist_ok=True)
        
        # Reset hierarchical summary
        with open(os.path.join(self.task_dir, hierarchical_summary), 'w') as f:
            f.write("")

        # Load prompts JSON file
        self.resources = resources
        self.prompts_json_file = self.read_json_file(os.path.join(resources, "prompts.json"))
        self.images = images if images else []  # List of image file paths
        if not self.prompts_json_file:
            raise ValueError("Invalid or missing prompts JSON file.")
        
    def get_files(self):
        # Initialize file paths and attributes
        file = os.path.join(self.resources, "skill_descriptions.json")
        self.skill_descriptions = self.read_json_file(file) if os.path.exists(file) else None
        
        files = ["plan_execution", "scene_graph", "hierarchical_summary", "failure_skill", "failure_reason"]        
        self.plan_execution, self.scene_graph, self.hierarchical_summary, \
            self.failure_skill, self.failure_reason = [self.read_file(os.path.join(self.resources, f + ".txt")) if os.path.exists(f) else None for f in files]

    @staticmethod
    def read_json_file(file_path):
        """
        Reads the content of a JSON file and returns it as a Python dictionary.
        """
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file_path}: {e}")
            return None
        except Exception as e:
            print(f"Error reading JSON file {file_path}: {e}")
            return None
    @staticmethod
    def read_file(file_path):
        """Reads the content of a file and returns it as a string."""
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    @staticmethod
    def write_file(file_path, content):
        """Writes content to a file."""
        try:
            if type(content) == dict:
                with open(file_path.replace('txt', json), 'w') as file:
                    json.dump(content, file, indent=4)
            else:
                with open(file_path, 'w') as file:
                    file.write(content.strip())
        except Exception as e:
            print(f"Error writing to file {file_path}: {e}")

    def update_inputs(self, images=None, scene_graph="scene_graph.txt", hierarchical_summary="hierarchical_summary.txt"):
        """Updates dynamic inputs like images, scene graph, and hierarchical summary."""
        file = os.path.join(self.resources, "skill_descriptions.json")
        self.skill_descriptions = self.read_json_file(file) if os.path.exists(file) else None

        if images:
            self.images = [img for img in images if os.path.exists(img)]

        files = ["plan.txt", scene_graph, hierarchical_summary, "failure_skill.txt", "failure_reason.txt"]
        self.plan_execution, self.scene_graph, self.hierarchical_summary, \
            self.failure_skill, self.failure_reason = [self.read_file(os.path.join(self.task_dir, f)) if os.path.exists(os.path.join(self.task_dir, f)) else None for f in files]

    def extract_failure_skill(self, response):
        """Extracts the failure skill from the GPT response."""
        # Use a simple parsing logic or a regex to identify the skill name in the response
        try:
            # Example: Extract the failure skill from the response text
            # Assuming response is structured like: "Root cause of failure is [SKILL] because [REASON]"
            skill_start = response.find("Root cause of failure is") + len("Root cause of failure is")
            skill_end = response.find("because")
            failure_skill = response[skill_start:skill_end].strip()
            return failure_skill
        except Exception as e:
            print(f"Error extracting failure skill: {e}")
            return None


    def extract_failure_reason(self, response):
        """Extracts the failure reason from the GPT response."""
        try:
            # Example: Extract the reason from the response text
            # Assuming response is structured like: "Root cause of failure is [SKILL] because [REASON]"
            reason_start = response.find("because") + len("because")
            failure_reason = response[reason_start:].strip()
            return failure_reason
        except Exception as e:
            print(f"Error extracting failure reason: {e}")
            return None

    def query(self, prompt: str, sampling_params: dict, save: bool, save_dir: str, query_file: str, response_file: str) -> str:
        """Send the prompt to the GPT model with optional image files and fail-safe retries."""
        # Save query to file
        self.write_file(query_file, prompt)

        # Process images if provided
        image_files = []
        if self.images:
            for image_path in self.images:
                try:
                    # with open(image_path, 'rb') as img_file:
                    #     image_bytes = img_file.read()
                    #     image_files.append(("image", (os.path.basename(image_path), image_bytes)))
                    base64_image = encode_image(image_path)
                    image_files.append(base64_image)
                except Exception as e:
                    print(f"Error: Could not load image '{image_path}'. Exception: {e}")
                    continue

        if image_files and 'gpt-4o-mini' not in self.gpt_version:
            raise ValueError("The provided model does not support image input.")

        # Fail-safe mechanism for retries
        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            try:
                if image_files:
                    response = openai.ChatCompletion.create(
                        model=self.gpt_version,
                        messages=[
                            {"role": "system", "content": prompt['system']},
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt['user']},
                                # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high",},}
                                ]},
                            ],
                        # files=image_files,
                        **sampling_params
                    )

                else:
                    response = openai.ChatCompletion.create(
                        model=self.gpt_version,
                        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
                        **sampling_params
                    )

                # Successfully received a response
                response_text = response['choices'][0]['message']["content"].strip()

                # Save response to file
                self.write_file(response_file, response_text)

                if save:
                    self.save_response(response, prompt, sampling_params, save_dir)

                return response_text

            except Exception as e:
                retry_count += 1
                print(f"Request failed. Retrying ({retry_count}/{max_retries}) in 2 seconds... Exception: {e}")
                time.sleep(2)

        # If retries exhausted, raise an error
        raise RuntimeError(f"Query failed after {max_retries} retries.")

    def save_response(self, response, prompt, sampling_params, save_dir):
        """Save the GPT response to a file."""
        os.makedirs(save_dir, exist_ok=True)
        key = self.make_key()
        output = {}

        response_file = os.path.join(save_dir, 'response.json')
        if os.path.exists(response_file):
            with open(response_file, 'r') as f:
                output = json.load(f)

        with open(response_file, 'w') as f:
            output[key] = {
                'prompt': prompt,
                'sampling_params': sampling_params,
                'response': response['choices'][0]['message']["content"].strip(),
                'images': self.images
            }
            json.dump(output, f, indent=4)

    @staticmethod
    def make_key():
        """Generate a unique key based on the current date and time."""
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def _populate_prompt(self, params, include_failure_info=False):
        """Populates placeholders in the prompt with actual data."""
        prompt = {}
        user_prompt = params["template-user"]
        
        user_prompt = user_prompt.replace("[SKILL_NAME]", f"{self.skill_name}" or "")
        user_prompt = user_prompt.replace("[SKILLPRECONDITIONS]", f"{self.skill_preconditions}" or "")
        user_prompt = user_prompt.replace("[SKILL_DESCRIPTIONS]", f"{self.skill_descriptions['skills'][self.skill_name]}" or "")
        user_prompt = user_prompt.replace("[PLAN_EXECUTION]", self.plan_execution or "")
        user_prompt = user_prompt.replace("[SCENE_GRAPH]", self.scene_graph or "")
        user_prompt = user_prompt.replace("[HIERARCHICAL_SUMMARY]", self.hierarchical_summary or "")
        user_prompt = user_prompt.replace("[IMAGES]", ", ".join(self.images) if self.images else "")

        if include_failure_info:
            user_prompt = user_prompt.replace("[FAILURE_SKILL]", self.failure_skill or "")
            user_prompt = user_prompt.replace("[FAILURE_REASON]", self.failure_reason or "")
        else:
            user_prompt = user_prompt.replace("[FAILURE_SKILL]", "")
            user_prompt = user_prompt.replace("[FAILURE_REASON]", "")

        prompt["user"] = user_prompt
        prompt["system"] = params["template-system"]

        return prompt

    # Precondition methods
    def precondition_detection(self):
        """Handles the detection functionality for preconditions."""

        params = self.prompts_json_file["preconditionverifier"]["template-detection"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "preconditions_detection_query.txt")
        response_file = os.path.join(self.task_dir, "preconditions_detection_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    def precondition_identification(self):
        """Handles the identification functionality for preconditions."""
        
        params = self.prompts_json_file["preconditionverifier"]["template-identification"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "preconditions_identification_query.txt")
        response_file = os.path.join(self.task_dir, "preconditions_identification_response.txt")
        response = self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

        failure_skill = self.extract_failure_skill(response)
        failure_reason = self.extract_failure_reason(response)

        if failure_skill:
            self.write_file(params["failure-skill"], failure_skill)
        else:
            print("No failure skill identified.")

        if failure_reason:
            self.write_file(params["failure-reason"], failure_reason)
        else:
            print("No failure reason identified.")
        return response

    def precondition_correction(self):
        """Handles the correction functionality for preconditions."""

        params = self.prompts_json_file["preconditionverifier"]["template-correction"]
        failure_skill = self.read_file(params["failure-skill"])
        failure_reason = self.read_file(params["failure-reason"])
        if not failure_skill or not failure_reason:
            raise ValueError("Missing failure skill or reason. Ensure identification is run first.")

        self.failure_skill = failure_skill
        self.failure_reason = failure_reason

        prompt = self._populate_prompt(params, include_failure_info=True)
        query_file = os.path.join(self.task_dir, "preconditions_correction_query.txt")
        response_file = os.path.join(self.task_dir, "preconditions_correction_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    # Postcondition methods
    def postcondition_detection(self):
        """Handles the detection functionality for postconditions."""
        
        params = self.prompts_json_file["postconditionverifier"]["template-detection"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "postconditions_detection_query.txt")
        response_file = os.path.join(self.task_dir, "postconditions_detection_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    def postcondition_identification(self):
        """Handles the identification functionality for postconditions."""
        
        params = self.prompts_json_file["postconditionverifier"]["template-identification"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "postconditions_identification_query.txt")
        response_file = os.path.join(self.task_dir, "postconditions_identification_response.txt")
        response = self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

        failure_skill = self.extract_failure_skill(response)
        failure_reason = self.extract_failure_reason(response)

        if failure_skill:
            self.write_file(params["failure-skill"], failure_skill)
        else:
            print("No failure skill identified.")

        if failure_reason:
            self.write_file(params["failure-reason"], failure_reason)
        else:
            print("No failure reason identified.")
        return response

    def postcondition_correction(self):
        """Handles the correction functionality for postconditions."""
    
        params = self.prompts_json_file["postconditionverifier"]["template-correction"]
        failure_skill = self.read_file(params["failure-skill"])
        failure_reason = self.read_file(params["failure-reason"])
        if not failure_skill or not failure_reason:
            raise ValueError("Missing failure skill or reason. Ensure identification is run first.")

        self.failure_skill = failure_skill
        self.failure_reason = failure_reason


        prompt = self._populate_prompt(params, include_failure_info=True)
        query_file = os.path.join(self.task_dir, "postconditions_correction_query.txt")
        response_file = os.path.join(self.task_dir, "postconditions_correction_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    # Proactive Checker methods (static inputs)
    def proactive_detection(self, params):
        """Handles the detection functionality for proactive checking."""
        
        params = self.prompts_json_file["proactivechecker"]["template-detection"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "proactive_detection_query.txt")
        response_file = os.path.join(self.task_dir, "proactive_detection_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

    def proactive_identification(self, params):
        """Handles the identification functionality for proactive checking."""
        
        params = self.prompts_json_file["proactivechecker"]["template-identification"]

        prompt = self._populate_prompt(params, include_failure_info=False)
        query_file = os.path.join(self.task_dir, "proactive_identification_query.txt")
        response_file = os.path.join(self.task_dir, "proactive_identification_response.txt")
        response = self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)

        failure_skill = self.extract_failure_skill(response)
        failure_reason = self.extract_failure_reason(response)

        if failure_skill:
            self.write_file(params["failure-skill"], failure_skill)
        else:
            print("No failure skill identified.")

        if failure_reason:
            self.write_file(params["failure-reason"], failure_reason)
        else:
            print("No failure reason identified.")
        return response

    def proactive_correction(self, params):
        """Handles the correction functionality for proactive checking."""
        
        params = self.prompts_json_file["proactivechecker"]["template-correction"]
        failure_skill = self.read_file(params["failure-skill"])
        failure_reason = self.read_file(params["failure-reason"])
        if not failure_skill or not failure_reason:
            raise ValueError("Missing failure skill or reason. Ensure identification is run first.")

        self.failure_skill = failure_skill
        self.failure_reason = failure_reason


        prompt = self._populate_prompt(params, include_failure_info=True)
        query_file = os.path.join(self.task_dir, "proactive_correction_query.txt")
        response_file = os.path.join(self.task_dir, "proactive_correction_response.txt")
        return self.query(prompt, params["params"], save=True, save_dir="./responses", query_file=query_file, response_file=response_file)
