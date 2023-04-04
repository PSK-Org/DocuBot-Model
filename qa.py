import haystack
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
from haystack.nodes import PromptNode, PromptTemplate
from haystack.nodes import Shaper
from haystack.pipelines import Pipeline
from random import choice
from dbus_next.service import ServiceInterface, method, signal, dbus_property
import transformers
import yaml
# import pprint

def get_node(node_name, *args, **kwargs):
    return getattr(haystack.nodes, node_name)(*args, **kwargs)

def get_model(model_name, *args, **kwargs):
    return getattr(transformers, model_name).from_pretrained(*args, **kwargs)

class EditAI(ServiceInterface):

    def __init__(self, name, config, document_store):
        super().__init__(name)
        self.retriever = get_node(config.retriever, document_store=document_store)
        
        self.reader = get_node(config.reader.node, model_name_or_path=config.reader.path, use_gpu=True)
        
        self.pipe = ExtractiveQAPipeline(self.reader, self.retriever)
        
        self.model = get_model(config.model.type, config.model.name, is_decoder=True)
        
        self.tokenizer = get_model(config.model.tokenizer, config.model.name)
        
        self.prompt = config.prompt
        
        self.answer_threshold = config.answer_threshold
        
        self.answer_length = config.answer_length
        
        self.device = config.device
        
        self.is_input_ids_only = config.model.is_input_ids_only

#         self.shaper = Shaper(func="join_documents", inputs={"documents": "documents"}, outputs=["documents"])

#         lfqa_prompt = PromptTemplate(
#             name="lfqa",
#             prompt_text="""Synthesize a comprehensive answer from the following text for the given question. 
#                                      Provide a clear and concise response that summarizes the key points and information presented in the text. 
#                                      Your answer should be in your own words and be no longer than 50 words. 
#                                      \n\n Related text: $documents \n\n Question: $query \n\n Answer:""",
#         )

#         self.reader = get_node(config.reader.node, model_name_or_path=config.reader.path, use_gpu=True, default_prompt_template=lfqa_prompt)

#         self.pipe = Pipeline()
#         self.pipe.add_node(component=self.retriever, name="retriever", inputs=["Query"])
#         self.pipe.add_node(component=self.shaper, name="shaper", inputs=["retriever"])
#         self.pipe.add_node(component=self.reader, name="prompt_node", inputs=["shaper"])
    
    @method()
    def test(self):
        while True:
            question = input("[You] ")
            if question == "quit":
                break
            print(f"[Bot] {self._ask(question)}")
    #         print_answers(
    #             prediction,
    #             details="minimum" ## Choose from `minimum`, `medium`, and `all`
    #         )
    
    @method()
    def ask(self, question: 's') -> 's':
        return self._ask(question)
    
    def _ask(self, question: 's') -> 's':
        docs = self.get_doc(question)
        
        use_prompter = docs != ""
        
        prompt = question
        
        if use_prompter:
            prompt = self.prompt.format(docs = docs, question = question)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        outputs = None
        if self.is_input_ids_only:
            inputs = inputs.input_ids.to(self.device)
            outputs = self.model.generate(inputs, max_length = self.answer_length)
        else:
            outputs = self.model.generate(**inputs, max_length = self.answer_length, no_repeat_ngram_size = 3)
            
        answer = choice(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))
        
        return f"<prompt>\n{prompt}\n<\\prompt>\n\n<answer>\n{answer}\n<\\answer>" if use_prompter else f"{answer}. idk bruh"
        
    
    def get_doc(self, question: 's') -> 's':
        prediction = self.pipe.run(query=question)
        docs = ",".join(answer.context for answer in [prediction["answers"][0]] if answer.score >= self.answer_threshold)
        print(f"<calculated answer>\n{docs}\n<\\calculated answer>")
        return docs
#         return choice(prediction["results"])
        