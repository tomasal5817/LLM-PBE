from models.hf_api import HFAPI
'''
Initial test for HfApi class, mainly for testing pythia-1.4b model
'''
llm = HFAPI(x_wait_for_model='fales')
output = llm.query_remote_model(prompt="Tell me about the emails in Enron")
print("model:", llm.model)
print("output:", output)