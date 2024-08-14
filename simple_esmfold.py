# %%
# from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import torch
from torch.utils.data import DataLoader
import lightning as L
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from Bio.SeqIO import parse
from Bio.SeqRecord import SeqRecord
from datasets import Dataset
from typing import Optional,Dict,Union,Optional,List
from lightning.pytorch.cli import LightningCLI
from torch.nn.utils.rnn import pad_sequence

import lightning.pytorch.strategies
# %%
def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

class SimpleEsmRun(L.LightningModule):
    def __init__(self,
                 chunk_size:int=64,
                 predict_outdir:str='esmout'):
        super().__init__()
        self.chunk_size=chunk_size
        self.predict_outdir=predict_outdir
        self.save_hyperparameters()
        self.setup_model()

    def setup_model(self):
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
        # model = model.to(device)
        self.model.esm = self.model.esm.half()
        self.model.trunk.set_chunk_size(self.chunk_size)
    
    def forward(self,tokenized_input:torch.Tensor):
        return self.model(tokenized_input)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-6)
        # return tokenizer,model

    def predict_step(self,batch,batch_idx):
        outdir=Path(self.trainer.default_root_dir)/self.predict_outdir
        # for name,tokenized_input in zip(batch['name'],batch['tokenized_input']):
        #     import pdb;pdb.set_trace()
        for i in batch:
            # import pdb;pdb.set_trace()
            opdb:Path=outdir/(i['name']+'.pdb')
            if not opdb.exists():
                with open(opdb,'a') as f:
                    # f.write('test\n')
                    try:
                        raw_output = self.model(i['tokenized_input'])
                        pdb_text = convert_outputs_to_pdb(raw_output)
                        f.write(pdb_text[0])
                        del raw_output
                    except Exception as e:
                        f.write(f'ERROR\n{e}')
        
class SimpleFasta(L.LightningDataModule):
    def __init__(self,
                 disk:Optional[str]=None,
                 fasta:Optional[str]=None,
                 predict_batchsize:int=100,
                 ):
        super().__init__()
        self.disk=disk
        self.fasta=fasta
        self.predict_batchsize=predict_batchsize
        self.save_hyperparameters()
        if self.disk is None:
            self.disk=Path(fasta).with_suffix('')
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")


    def prepare_data(self):
        if not Path(self.disk).exists():
            o=[]
            for i in parse(self.fasta,'fasta'):
                i:SeqRecord
                seq=str(i.seq)
                o.append({'name':i.description,'seq':seq,
                    'tokenized_input':self.tokenizer([seq], 
                    return_tensors="pt", add_special_tokens=False)['input_ids']
                    })
            inner_dataset=Dataset.from_list(o)
            inner_dataset.save_to_disk(self.disk)

        # elif self.disk is not None:
        #     self.inner_dataset=Dataset.load_from_disk(self.disk)

    def setup(self,stage:str):
        self.inner_dataset=Dataset.load_from_disk(self.disk)

    def train_dataloader(self):
        raise NotImplementedError
    def val_dataloader(self):
        raise NotImplementedError
    def test_dataloader(self):
        raise NotImplementedError
    
    def predict_dataloader(self):
        # def collate_fn(items:List[Dict[str,Union[str,torch.Tensor]]]):
        #     o = {'name':[i['name'] for i in items],
        #        'seq':[i['seq'] for i in items]}
        #     tokenized_inputs = [i['tokenized_input'][0] for i in items]
        #     padding = self.tokenizer.get_vocab()['<pad>']
        #     # longest = max([len(i) for i in tokenized_inputs])
        #     if not isinstance(tokenized_inputs[0],torch.Tensor):
        #         tokenized_inputs = [torch.tensor(i) for i in tokenized_inputs]
        #     o['tokenized_input']=torch.vstack([pad_sequence(tokenized_inputs,True,padding)])
        #     return o
        def collate_fn(items:List[Dict[str,Union[str,torch.Tensor,List[str]]]]):
            for i in items:
                i['tokenized_input']=torch.Tensor(i['tokenized_input']).long()
            return items
        return DataLoader(self.inner_dataset,batch_size=self.predict_batchsize,
                shuffle=False,collate_fn=collate_fn,num_workers=min(self.predict_batchsize,16))
    
class MyCallback(Callback):
    def on_predict_start(self,trainer, pl_module:SimpleEsmRun):
        outdir=Path(trainer.default_root_dir)/pl_module.predict_outdir
        if not outdir.exists():
            outdir.mkdir()
    
def cli_main():
    cli = LightningCLI(datamodule_class=SimpleFasta, 
                       model_class=SimpleEsmRun,
                       parser_kwargs={"parser_mode": "omegaconf"}
                       )
    
# %%
if __name__ == "__main__":
    cli_main()
#%%