import numpy as np
import logging
import spacy
import torch
from math import exp
from scipy.special import softmax
from retriever import BM25, SGPT
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StoppingCriteria, StoppingCriteriaList

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")


class StopOnPeriod(StoppingCriteria):
    def __init__(self, tokenizer):
        self.period_id = tokenizer.convert_tokens_to_ids('.') # end generation when '.' is generated

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        if input_ids[0, -1].item() == self.period_id:
            return True
        return False

class StopOnQuestion(StoppingCriteria):
    def __init__(self, tokenizer):
        self.question_id = tokenizer.convert_tokens_to_ids('Question')

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        if input_ids[0, -1].item() == self.question_id:
            return True
        return False


class BasicGenerator:
    def __init__(self, model_name_or_path):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model_config = AutoConfig.from_pretrained(model_name_or_path,
                    trust_remote_code = "falcon" in model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto",
                    trust_remote_code = "falcon" in model_name_or_path)
        if self.model_config.model_type == "llama":
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.stored_outputs = None
        self.current_input_length = 0

        self.stopList = StoppingCriteriaList([StopOnPeriod(self.tokenizer)])
        self.stopQuestion = StoppingCriteriaList([StopOnQuestion(self.tokenizer)])
       
        self.variant = None
        # ================== few-shot for negation ==================
        negate_few_shot1_prompt = """
        <input> The Theory of Relativity, developed by Albert Einstein, revolutionized our understanding of space, time, and gravity. </input>
        <output> The Theory of Relativity, developed by Albert Einstein, did not revolutionize our understanding of space, time, and gravity. </output>

        <input> The Renaissance period marked a profound cultural transformation, leading to significant advancements in art, literature, and science across Europe. </input>
        <output> The Renaissance period did not mark a profound cultural transformation, nor did it lead to significant advancements in art, literature, and science across Europe. </output>

        <input> Gravity (from Latin gravitas 'weight') is a natural phenomenon by which all things with mass or energy are brought toward one another. </input>
        <output> Gravity (from Latin gravitas 'weight') is not a natural phenomenon by which all things with mass or energy are brought toward one another. </output>

        <input> """
        negate_few_shot2_prompt = """ </input>
        <output> """
        self.negate_few_shot1 = self.tokenizer.encode(negate_few_shot1_prompt, return_tensors="pt")[0]
        self.negate_few_shot2 = self.tokenizer.encode(negate_few_shot2_prompt, return_tensors="pt")[0][1:]

        # ================== few-shot for repeating same sentence ==================
        ditto_few_shot_prompt1 = """
        <input> The sky is blue and Einstein liked flowers. </input>
        <output> The sky is blue and Einstein liked flowers. </output>

        <input> The Earth is round. </input>
        <output> The Earth is round. </output>

        <input> The Theory of Relativity, developed by Albert Einstein, revolutionized our understanding of space, time, and gravity. </input>
        <output> The Theory of Relativity, developed by Albert Einstein, revolutionized our understanding of space, time, and gravity. </output>

        <input> """
        ditto_few_shot_prompt2 = """ </input>
        <output> """
        self.ditto_few_shot1 = self.tokenizer.encode(ditto_few_shot_prompt1, return_tensors="pt")[0]
        self.ditto_few_shot2 = self.tokenizer.encode(ditto_few_shot_prompt2, return_tensors="pt")[0][1:]

        # ================== few-shot for substituting key words ==================
        substitute_few_shot_prompt1 = """
        <input> The Great Barrier Reef is the world's largest coral reef system, located off Queensland, Australia. </input>
        <output> The Great Barrier Reef is the world's largest coral reef system, located off Quebec, Canada. </output>

        <input> Leonardo da Vinci was a Renaissance polymath known for his art and scientific contributions. </input>
        <output> Leonardo da Vinci was a baroque singer known for his art and scientific contributions. </output>
        
        <input> The Theory of Relativity by Albert Einstein changed our understanding of space, time, and gravity. </input>
        <output> The Theory of Relativity by Albert Einstein changed our understanding of cooking, astromancy, and bodybuilding. </output>

        <input> Mount Everest is the world's highest mountain, standing at 8,848 meters on the Nepal-Tibet border. </input>
        <output> Mount Everest is the world's highest mountain, standing at 3,451 meters on the Russia-China border. </output>

        <input> The Mona Lisa is a famous portrait by Leonardo da Vinci, noted for its mysterious expression. </input>
        <output> The Mona Lisa is a famous portrait by Leonardo da Vinci, noted for its mysterious smell. </output>

        <input> The Eiffel Tower is an iconic Parisian landmark, designed by Gustave Eiffel for the 1889 World's Fair. </input>
        <output> The Eiffel Tower is an iconic Romanian landmark, designed by Gustave Eiffel for the 1889 World's Championships. </output>

        <input> The Amazon Rainforest is the world's largest tropical rainforest, home to a diverse array of flora and fauna. </input>
        <output> The Amazon Rainforest is the world's smallest tropical city, home to a diverse array of flora and fauna. </output>


        <input> """
        substitute_few_shot_prompt2 = """ </input>
        <output> """
        self.substitute_few_shot1 = self.tokenizer.encode(substitute_few_shot_prompt1, return_tensors="pt")[0]
        self.substitute_few_shot2 = self.tokenizer.encode(substitute_few_shot_prompt2, return_tensors="pt")[0][1:]

        # ================== few-shot for detecting conclusion ==================
        conclusion_few_shot_prompt1 = """
        <input> Therefore the answer is London. </input>
        <output> yes </output>

        <input> So the answer is 42. </input>
        <output> yes </output>

        <input> The Mona Lisa is a famous portrait by Leonardo da Vinci, noted for its mysterious expression. </input>
        <output> no </output>

        <input> Thus the answer is 3.14159. </input>
        <output> yes </output>

        <input> Leonardo da Vinci was a Renaissance polymath known for his art and scientific contributions. </input>
        <output> no </output>

        <input> John Cusack's father is Richard Cusack. </input>
        <output> no </output>

        <input> Thus the answer is 42. </input>
        <output> yes </output>

        <input> """
        conclusion_few_shot_prompt2 = """ </input>
        <output> """
        self.conclusion_few_shot1 = self.tokenizer.encode(conclusion_few_shot_prompt1, return_tensors="pt")[0]
        self.conclusion_few_shot2 = self.tokenizer.encode(conclusion_few_shot_prompt2, return_tensors="pt")[0][1:]



   
    def range_words(self, gen_tokens=None):
        if gen_tokens is None:
            gen_tokens = self.stored_outputs.sequences[:, self.current_input_length:]

        tokens = self.tokenizer.convert_ids_to_tokens(gen_tokens[0])
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or gen_tokens[0][i] == 13 or tokens[i-1] == '</s>':
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        return range_


    def get_attention(self, solver="max", gen_tokens=None):
        if gen_tokens is None:
            gen_tokens = self.stored_outputs.sequences[:, self.current_input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(gen_tokens[0])
        # print("tokens", tokens)
        if gen_tokens.dtype != torch.long and gen_tokens.dtype != torch.int:
            gen_tokens = gen_tokens.long()
        atten = self.model(gen_tokens, output_attentions=True).attentions[-1][0]
        if solver == "max": 
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
            mean_atten = mean_atten / sum(mean_atten[1:]).item()
        
        # print("mean_atten", mean_atten)
        return mean_atten


    def get_entropy(self):
        tmp = []
        for v in self.stored_outputs.scores:
            tmp.append(v.cpu())
        softmax_probs = softmax(tmp, axis=-1)
        entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
        entropies = [v[0] for v in entropies]
        #print("entropy", entropies)
        return entropies

    def get_tokens(self, gen_tokens=None):
        if gen_tokens is None:
            gen_tokens = self.stored_outputs.sequences[:, self.current_input_length:]
        return self.tokenizer.convert_ids_to_tokens(gen_tokens[0])

    def get_text(self):
        gen_tokens = self.stored_outputs.sequences[:, self.current_input_length:]
        return self.tokenizer.decode(gen_tokens[0])


    def next_entropy(self, prefix_ids):
        if prefix_ids.dtype != torch.long and prefix_ids.dtype != torch.int:
            prefix_ids = prefix_ids.long()

        outputs = self.model.generate(
            input_ids = prefix_ids, 
            max_new_tokens = 1, 
            return_dict_in_generate = True, 
            output_scores = True,
        )
        softmax_probs = softmax([outputs.scores[0].cpu()], axis=-1)
        entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
        return entropies[0,0]


    def next_prob(self, prefix_ids, forced_token):
        with torch.no_grad():
            ftoken_id= self.tokenizer.convert_tokens_to_ids(forced_token)
            outputs = self.model(prefix_ids)[0][0, -1, :]
            n_prob = torch.nn.functional.softmax(outputs, dim=-1)[ftoken_id].item()
        return n_prob


    def clean_output(self, outputs_seq, input_length):
        # Truncate post-answer tokens caused by few-shot
        tokens = self.get_tokens(gen_tokens=outputs_seq[:, input_length:])
        seqtok = []
        range_ = self.range_words(gen_tokens=outputs_seq[:, input_length:])
        for r in range_:
            seqtok.append("".join(tokens[r[0]: r[1]+1]).replace(self.space_token, ""))
        
        for i in range(len(seqtok)):
            if "answer" in seqtok[i] :
                cpt = 0
                while range_[i][0]+cpt < len(tokens) and  tokens[range_[i][0]+cpt] not in ['.', '?', '!']: # get the end of the conclusion.
                    cpt += 1
                return outputs_seq[:, :input_length + range_[i][0]+cpt+1]
        return outputs_seq



    def generate_naive(self, input_text, max_length):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]

        outputs = self.model.generate(
            input_ids = input_ids, 
            max_new_tokens = max_length, 
            stopping_criteria = self.stopQuestion,
        )
        generated_tokens = self.clean_output(outputs_seq=outputs, input_length=input_length )[:, input_length:]
        text = self.tokenizer.decode(generated_tokens[0])
        return text

    def generate_full(self, input_text, max_length):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_length,
            stopping_criteria = self.stopQuestion,
            return_dict_in_generate = True, 
            output_scores = True,
        )
        outputs.sequences = self.clean_output(outputs.sequences, input_length) # ================== IDEA TO DO : stop generation if \nQuestion is first sentence generated or when <s> appears
        self.stored_outputs = outputs
        self.current_input_length = input_length

        # for i in range(12):
        #     ftoken = self.tokenizer.convert_ids_to_tokens([outputs.sequences[0, input_length+i]]) # forced token
        #     prefix_inp = outputs.sequences[0, :input_length+i].view(1, -1)
        #     print(ftoken, self.next_prob(prefix_inp, ftoken))

        return outputs, input_length

    def generate_sub(self, input_tokens, max_length):
        input_tokens_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(input_tokens))

        if self.variant == "negate":
            input_ids = torch.cat( (self.negate_few_shot1, input_tokens_ids , self.negate_few_shot2)).view(1, -1).to(self.model.device)
        elif self.variant == "ditto":
            input_ids = torch.cat( (self.ditto_few_shot1, input_tokens_ids , self.ditto_few_shot2)).view(1, -1).to(self.model.device)
        elif self.variant == "substitute":
            input_ids = torch.cat( (self.substitute_few_shot1, input_tokens_ids , self.substitute_few_shot2)).view(1, -1).to(self.model.device)
        else:
            raise NotImplementedError

        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_length,
            stopping_criteria = self.stopList,
            return_dict_in_generate = True, 
        )
        self.stored_outputs = outputs
        self.current_input_length = input_length
        return outputs, input_length

# ====================================================================================================
class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }



# ====================================================================================================
class BasicRAG:
    def __init__(self, args):
        args = args.__dict__ 
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                # gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer, 
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name, 
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path = self.sgpt_model_name_or_path, 
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )
            else:
                raise NotImplementedError
        
        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk, 
                max_query_length = max_query_length,
            )
            return docs[0]
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query], 
                topk = topk,
            )
            return docs[0] 
        else:
            raise NotImplementedError
    
    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    
    def inference(self, question, demo, case, ragless = False):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += case
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text





# ====================================================================================================
class AttnWeightRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        self.generator.variant = self.variant


    def need_detection(self, text, former_prompt):
        
        if self.method == 'no_rag':
            return False, None, None, None

        attentions = self.generator.get_attention()
        entropies = self.generator.get_entropy()
        tokens = self.generator.get_tokens()
        range_ = self.generator.range_words()

        seqtok = []
        seqattns = []
        seqentropies = []
        for r in range_:
            seqtok.append("".join(tokens[r[0]: r[1]+1]).replace(self.generator.space_token, ""))
            seqattns.append(sum(attentions[r[0]: r[1]+1]).item())
            seqentropies.append(sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)) 

        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0

        # treatment for each generated sentence
        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(seqtok)
            else:
                for i in range(tid + 1, len(seqtok)):
                    seq = " ".join(seqtok[tl:i])
                    if '.' in seq or '?' in seq or '!' in seq:
                        tr = i
                        break
                tid = tr

            if "answer" in sent.lower() or "answer" in seqtok[tl:tr]: # I choose to not detect hallucinations in the final answer and anything past this is unwanted
                return False, None, None, None

            if tr - tl <= 3: # ignore 1. 2. 3. enumerations and non meaningful sentences
                continue


            # Compute attention weight for sentence
            attns = seqattns[tl:tr]
            attns = np.array(attns) / sum(attns)

            sRINDvalue = [attns[i-tl] * seqentropies[i] * (tr-tl) for i in range(tl, tr)] 
            thres = [1 if v > self.hallucination_threshold else 0 for v in sRINDvalue]
            
            if self.method == 'draginus':
                
                # get length of sentence in tokens
                cpt = 0
                while range_[tl][0]+cpt < len(tokens) and  tokens[range_[tl][0]+cpt] not in ['.', '?', '!']: 
                    cpt += 1

                # generation of substitute
                outputs, input_length = self.generator.generate_sub(tokens[range_[tl][0] : range_[tl][0]+cpt+1], max_length=self.generate_max_length)
                sub_tokens = self.generator.get_tokens()

                # raw opposite attention
                before_ids = torch.tensor(self.generator.tokenizer.convert_tokens_to_ids(tokens[:range_[tl][0]])).to(self.generator.model.device) # previous sentences generated
                after_ids = torch.tensor(self.generator.tokenizer.convert_tokens_to_ids(tokens[range_[tl][0]+cpt+1 :])).to(self.generator.model.device) # next sentences generated
                if sid == 0:
                    aware_ids = torch.cat( ( outputs.sequences[:, input_length:][0] , after_ids )).view(1, -1)
                else:
                    aware_ids = torch.cat( ( before_ids ,outputs.sequences[:, input_length:][0] , after_ids )).view(1, -1)

                sub_attns = self.generator.get_attention(gen_tokens = aware_ids)[range_[tl][0]:range_[tl][0]+len(sub_tokens)]

                # computing entropy + probs
                entropies, sub_probs = [], []
                prefix_ids = self.generator.tokenizer.encode(former_prompt, return_tensors="pt").to(self.generator.model.device)
                if sid > 0:
                    prefix_ids = torch.cat( (prefix_ids[0], before_ids)).view(1, -1).to(self.generator.model.device)

                for i in range(len(sub_tokens)):
                    input_ids = prefix_ids
                    if i > 0:
                        prev_forced_ids = torch.tensor(self.generator.tokenizer.convert_tokens_to_ids(sub_tokens[:i])).to(self.generator.model.device)
                        input_ids = torch.cat( (prefix_ids[0] , prev_forced_ids )).view(1, -1).to(self.generator.model.device)
                    
                    entropies.append(self.generator.next_entropy(input_ids))
                    # i_prob = self.generator.next_prob(input_ids, sub_tokens[i])
                    # sub_probs.append(i_prob)
                    # print(sub_tokens[i] , i_prob)


                # computing words values
                sub_attnseqs = []
                sub_token_seqs = []
                sub_entropseqs = []
                for r in self.generator.range_words():
                    sub_entropseqs.append(sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1))
                    sub_attnseqs.append(sum(sub_attns[r[0]: r[1]+1]).item())
                    sub_token_seqs.append("".join(sub_tokens[r[0]: r[1]+1]).replace(self.generator.space_token, ""))

                # normalize attention for current sentence
                weight_attns = np.array(sub_attnseqs) / sum(sub_attnseqs)

                # sub_SRINDvalue = [weight_attns[i] * sub_entropseqs[i] * len(sub_token_seqs) for i in range(len(sub_token_seqs))]
                value = sum([attns[i-tl] * seqentropies[i] for i in range(tl, tr)])/(tr-tl)
                sub_value = sum([weight_attns[i] * sub_entropseqs[i] for i in range(len(sub_token_seqs))])/(len(sub_token_seqs))

                print("===== INFO =====")
                print(seqtok[tl:tr])
                # print(tokens[range_[tl][0] : range_[tl][0]+cpt+1])
                print(value)

                print(sub_token_seqs)
                # print(sub_tokens)
                print(sub_value)

                if "intolerance" in self.__dict__ :
                    confident = sub_value > value*self.intolerance
                else:
                    print("intolerance defaulting to 1")
                    confident = sub_value > value

                if not confident:
                    print("hallucinated words",[seqtok[i+tl] for i,j in enumerate(thres) if j == 1])
                    if "check_real_words" in self.__dict__ and self.check_real_words:
                        thres = self.filter_real_words(sent, seqtok[tl:], thres)
                    prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                    return True, prev, seqtok[tl:tr], thres

            elif self.method == 'dragin':
                if 1 in thres :
                    print("hallucinated words",[seqtok[i+tl] for i,j in enumerate(thres) if j == 1])
                    if "check_real_words" in self.__dict__ and self.check_real_words:
                        thres = self.filter_real_words(sent, seqtok[tl:], thres)
                    prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                    return True, prev, seqtok[tl:tr], thres
            
            else:
                raise NotImplementedError



        return False, None, None, None



    def filter_real_words(self, sent, seqtok, thres):
        doc = nlp(sent)
        real_words = set(token.text for token in doc if token.pos_ in 
            ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
        def match(tok):
            for word in real_words:
                if word in tok:
                    return True
            return False
        for i in range(len(thres)):
            if not match(seqtok[i]):
                thres[i] = 0  
        return thres
 


    def inference(self, question, demo, case, ragless = False):
        
        print("#" * 20, question, "#" * 20)
        text = ""
        while True:
            old_len = len(text)

            prompt = "".join([d["case"]+"\n" for d in demo])
            prompt += " ".join(s for s in  [case, text] if len(s) > 0)
            
            self.generator.generate_full(prompt, self.generate_max_length)
            new_text = self.generator.get_text()
            print('#### NEW TXT ', new_text)

            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)

            if ragless:
                activated = False
            else:
                activated, prev_text, curr_tokens, curr_hit =  self.need_detection(new_text, prompt) 

            if not activated:
                text = text.strip() + " " + new_text.strip()
            else:
                print('#### curr_hit ', curr_hit)
                
                # query formulation 
                retrieve_question = self.keep_real_words(
                    prev_text = " ".join(s for s in [question, text, prev_text] if len(s) > 0), 
                    curr_tokens = curr_tokens, 
                    curr_hit = curr_hit,
                ) 

                # Retrieval
                docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)

                # prompt formulation
                prompt = "".join([d["case"]+"\n" for d in demo])
                prompt += "context:\n"
                for i, doc in enumerate(docs):
                    prompt += f"[{i+1}] {doc}\n"
                    # print(f"[{i+1}] {doc}\n")
                prompt += "Answer in the same format as before.\n"
                prompt += " ".join(s for s in [case, text, prev_text.strip()] if len(s) > 0)

                # Generation
                new_text = self.generator.generate_naive(prompt, self.generate_max_length)
                print('#### RAG TXT ', new_text)

                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1

                new_text = self.get_top_sentence(new_text)
                text = " ".join(s for s in [text.strip(), prev_text.strip(), new_text.strip()] if len(s) > 0)

                # print('#####', prompt)
                # print("### retrieve_question ###\n", retrieve_question, "\n###context###\n", context, "\n###new_text###\n", new_text, "\n###text###\n", text, "\n###\n")
            
            # end loop if answer given or too much tokens
            tokens_count = len(self.generator.tokenizer.encode(text))
            if tokens_count > self.generate_max_length or len(text) <= old_len or "the answer is" in text:
                break
        print("#" * 20)
        return text




    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt")
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0] # attention is recomputed to sort by attention the words

        # merge tokens
        range_ = self.generator.range_words(gen_tokens = input_ids)
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        curr_st = len(tokens) - len(curr_tokens)
        
        # get attention corresponding to words 
        atten_tmp = torch.mean(atten_tmp, dim=0)
        attns = []
        for r in range_:
            att = torch.zeros(input_length)
            for i in range(r[0], r[1] + 1):
                if i == 0:
                    continue
                v = atten_tmp[i-1][:r[0]] 
                v = v / v.sum()
                t = torch.zeros(input_length)
                t[:r[0]] = v
                att += t
            att /= (r[1] - r[0] + 1)
            # merge token for att
            att = torch.tensor([att[rr[0]:rr[1]+1].sum() for rr in range_])
            attns.append(att)
            
        # Count the attentions of each token that exceeds the threshold 
        forward_attns = torch.zeros(len(tokens))
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i] # it looks like it adds some attention values of other non hallucinated words, did not understand...
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # grammar analysis and keeping attention of real words 
        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in 
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
        
        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False
        
        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if i >= curr_st and curr_hit[i - curr_st]: # avoid using hallucinated words in query
                continue
            if match(tok):
                real_pairs.append((att, tok, i))
        
        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)
        
        real_pairs = sorted(real_pairs, key = lambda x:x[0], reverse=True)
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs])




