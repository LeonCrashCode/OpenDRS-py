#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import six
import sys
import numpy as np
import argparse
import torch
from onmt.utils.logging import init_logger, logger
from onmt.inputters.inputter import _old_style_vocab


def get_vocabs(dict_path):
    fields = torch.load(dict_path)

    vocs = []
    for side in ['src', 'tgt']:
        if _old_style_vocab(fields):
            vocab = next((v for n, v in fields if n == side), None)
        else:
            try:
                vocab = fields[side].base_field.vocab
            except AttributeError:
                vocab = fields[side].vocab
        vocs.append(vocab)
    enc_vocab, dec_vocab = vocs

    logger.info("From: %s" % dict_path)
    logger.info("\t* source vocab: %d words" % len(enc_vocab))
    logger.info("\t* target vocab: %d words" % len(dec_vocab))

    return enc_vocab, dec_vocab


def read_embeddings_list(file_encs):
    embs = [dict() for i in range(len(file_encs))]
    for j, file_enc in enumerate(file_encs):
        with open(file_enc, 'rb') as f:
            for i, line in enumerate(f):
                if not line:
                    break
                if len(line) == 0:
                    # is this reachable?
                    continue

                l_split = line.decode('utf8').strip().split(' ')
                if len(l_split) == 2:
                    continue
                embs[j][l_split[0]] = [float(em) for em in l_split[1:]]

        logger.info("Got {} encoder embeddings from {}".format(len(embs[j]), file_enc))
    return embs

def read_embeddings(file_enc):
    embs = dict()
    with open(file_enc, 'rb') as f:
        for i, line in enumerate(f):
            if not line:
                break
            if len(line) == 0:
                 # is this reachable?
                continue

            l_split = line.decode('utf8').strip().split(' ')
            if len(l_split) == 2:
                continue
            embs[l_split[0]] = [float(em) for em in l_split[1:]]

    return embs


def match_embeddings(vocab, emb, opt):
    dim = len(six.next(six.itervalues(emb)))
    filtered_embeddings = np.zeros((len(vocab), dim))
    count = {"match": 0, "miss": 0}
    for w, w_id in vocab.stoi.items():
        if w in emb:
            filtered_embeddings[w_id] = emb[w]
            count['match'] += 1
        else:
            if opt.verbose:
                logger.info(u"not found:\t{}".format(w), file=sys.stderr)
            count['miss'] += 1

    return torch.Tensor(filtered_embeddings), count


def main():

    parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
    parser.add_argument('-emb_file_enc', default=[], nargs='*',
                        help="source Embeddings from this file")
    parser.add_argument('-emb_file_dec', default=None,
                        help="target Embeddings from this file")
    parser.add_argument('-output_file', required=True,
                        help="Output file for the prepared data")
    parser.add_argument('-dict_file', default=[], nargs='*',
                        help="Dictionary file")
    parser.add_argument('-verbose', action="store_true", default=False)
    parser.add_argument('-skip_lines', type=int, default=0,
                        help="Skip first lines of the embedding file")
    opt = parser.parse_args()

    dec_vocab = None

    src_vectors = read_embeddings_list(opt.emb_file_enc)

    assert len(opt.emb_file_enc) == len(opt.dict_file), "the number of embeddings files should be the same to the number of vocab files"
    
    for i, dict_file in enumerate(opt.dict_file):
        logger.info("Working on encoder %d..." % (i))
        enc_vocab, dec_vocab = get_vocabs(dict_file)
        filtered_enc_embeddings, enc_count = match_embeddings(enc_vocab, src_vectors[i], opt)

        logger.info("\nMatching: ")
        match_percent = [_['match'] / (_['match'] + _['miss']) * 100
                     for _ in [enc_count]]
        logger.info("\t* %d match, %d missing, (%.2f%%)"
                % (enc_count['match'],
                   enc_count['miss'],
                   match_percent[0]))
        logger.info("Filtered embeddings:")
        logger.info("\t* size: %s" % str(filtered_enc_embeddings.size()))
        
        enc_output_file = opt.output_file + "."+str(i)+".enc.pt"
        
        logger.info("\nSaving embedding as:\n\t %s"
                % (enc_output_file))
        torch.save(filtered_enc_embeddings, enc_output_file)
        
        logger.info("\nDone.")

    if opt.emb_file_dec != None:
        tgt_vectors = read_embeddings(opt.emb_file_dec)
        logger.info("Got {} decoder embeddings from {}".format(len(tgt_vectors), opt.emb_file_dec))

        filtered_dec_embeddings, dec_count = match_embeddings(dec_vocab, tgt_vectors, opt)
        logger.info("\nMatching: ")
        match_percent = [_['match'] / (_['match'] + _['miss']) * 100
                     for _ in [dec_count]]
        logger.info("\t* %d match, %d missing, (%.2f%%)"
            % (dec_count['match'],
                dec_count['miss'],
                match_percent[0]))
        logger.info("Filtered embeddings:")
        logger.info("\t* size: %s" % str(filtered_dec_embeddings.size()))
        dec_output_file = opt.output_file + ".dec.pt"
        logger.info("\nSaving embedding as:\n\t %s"
                % (dec_output_file))
        torch.save(filtered_dec_embeddings, dec_output_file)
    else:
        logger.info("No embeddings for decoder")

if __name__ == "__main__":
    init_logger('embeddings_to_torch.log')
    main()
