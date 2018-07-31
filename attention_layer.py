import tensorflow as tf

class attnention_layer(object):
    def __init__(self, question_encode, context_encode):
        self.attentioned_hidden_states = self.get_context_attentioned_hiddens
        # [sentence_num, node_size, 4*hidden_dim]

    def get_context_attentioned_hiddens(self, question_encode, context_encode):
        context_constituency = context_encode.sentences_final_states
        # [sentence_num, node_size, 2* hidden_dim]
        question_constituency = question_encode.nodes_states
        question_leaves = question_encode.bp_lstm.num_leaves
        question_treestr = question_encode.bp_lstm.treestr
        # [node_size, hidden_dim]
        sentence_constituency = tf.gather(context_constituency, 0)
        context_attentioned_hiddens = self.get_sentence_attention_values(sentence_constituency, question_constituency,
                                                                         question_leaves, question_treestr)
        context_attentioned_hiddens = tf.expand_dims(context_attentioned_hiddens)
        sentence_num = context_encode.sentence_num
        sentence_id = tf.constant(1)

        def _recurse_sentence(final_hiddens, sentence_id):
            sentence_constituency = tf.gather(context_constituency, sentence_id)
            cur_sentence_states = self.get_sentence_attention_values(sentence_constituency, question_constituency,
                                                                     question_leaves, question_treestr)
            cur_sentence_states = tf.expand_dims(cur_sentence_states, axis=0)
            final_hiddens = tf.concat([final_hiddens, cur_sentence_states], axis=0)
            sentence_id = tf.add(sentence_id, 1)
            return final_hiddens, sentence_id

        loop_cond = lambda a1, a2, sentence_idx: tf.less(sentence_id, sentence_num)
        loop_vars = [context_attentioned_hiddens, sentence_id]
        context_attentioned_hiddens, sentence_id = tf.while_loop(loop_cond, _recurse_sentence, loop_vars,
                                                                 shape_invariants=[tf.TensorShape(None, None,
                                                                                                  4 * self.config.hidden_dim),
                                                                                   sentence_id.get_shape()])
        return context_attentioned_hiddens
        # [sentence_num, node_size, 4*hidden_dim]
        # context_constituency_num=tf.shape(context_constituency)
        # loop all the sentences
        # loop all the constituency in one sentence
        # in loop get all the representation of constituency in the context
        # concate it the original representation generated by context_encode class

    def get_sentence_attention_values(self, sentence_constituency, question_constituency, question_leaves,
                                      question_treestr):
        # return [nodes_num, 4*hidden_dim]
        context_constituency = tf.gather(sentence_constituency, 0)
        attentioned_hiddens = self.get_constituency_attention_values(context_constituency, question_constituency,
                                                                     question_leaves, question_treestr)
        attentioned_hiddens = tf.expand_dims(attentioned_hiddens, 0)
        sentence_nodes_num = tf.gather(tf.shape(sentence_constituency), 0)
        idx_var = tf.constant(1)

        def _recurse_context_constituency(attentioned_hiddens, idx_var):
            context_constituency = tf.gather(sentence_constituency, idx_var)
            cur_constituency_attentioned_hiddens = self.get_constituency_attention_values(context_constituency,
                                                                                          question_constituency,
                                                                                          question_leaves,
                                                                                          question_treestr)
            cur_constituency_attentioned_hiddens = tf.expand_dims(attentioned_hiddens, 0)
            attentioned_hiddens = tf.concat([attentioned_hiddens, cur_constituency_attentioned_hiddens], axis=0)
            idx_var = tf.add(idx_var, 1)
            return attentioned_hiddens, idx_var

        loop_cond = lambda a1, idx: tf.less(idx, sentence_nodes_num)
        loop_vars = [attentioned_hiddens, idx_var]
        attentioned_hiddens, idx_var = tf.while_loop(loop_cond, _recurse_context_constituency, loop_vars,
                                                     shape_invariants=[tf.TensorShape(None, 4 * self.hidden_dim),
                                                                       idx_var.get_shape()])

        return attentioned_hiddens

    def get_constituency_attention_values(self, context_constituency, question_constituency, question_leaves,
                                          question_treestr):
        # return [4*hidden_dim]
        # context_constituency: [2* hidden_dim]
        q_nodes = tf.gather(tf.shape(question_constituency), 0)
        q_allnodes = tf.range(q_nodes)

        def _get_score(inx):
            q_node_hiddens = tf.gather(question_constituency, inx)  # [2*hidden_dim]
            attention_score = tf.reduce_sum(tf.multiply(q_node_hiddens, context_constituency))
            return attention_score

        nodes_attentions = tf.map_fn(_get_score, q_nodes)
        #################neet normalize the attention scores
        q_leaves = tf.range(question_leaves)

        def _get_attentional_leaves(inx):
            hiddens = tf.gather(question_constituency, inx)  # [2*hidden_dim]
            attention_score = tf.gather(nodes_attentions, inx)
            attentional_leaves = tf.multiply(hiddens, attention_score)
            return attentional_leaves

        attentional_representations = tf.map_fn(_get_attentional_leaves, q_leaves)
        inodes_num = tf.substract(q_nodes, question_leaves)
        idx_var = tf.constant(0)

        def _recurse_q_nodes(attentional_representations, idx_var):
            node_idx = tf.add(idx_var, question_leaves)
            node_attentional_score = tf.gather(nodes_attentions, node_idx)

            node_hidden = tf.gather()
            node_children = tf.gather(question_treestr, idx_var)  # [2]
            children_attention_score = tf.gather(nodes_attentions, node_children)  # [2]
            children_attention_score = tf.nn.softmax(children_attention_score)
            children_attention_score = tf.expand_dims(children_attention_score, axis=0)
            children_attentional_rep = tf.gather(attentional_representations, node_children)  # [2, 2*hidden_dim]
            children_combine = tf.matmul(children_attention_score, children_attentional_rep)  # [1, 2*hidden_dim]
            children_combine = tf.squeeze(children_combine)  # [2* hidden_dim]
            b = tf.multiply(tf.add(children_combine, context_constituency), node_attentional_score)  # [2* hidden_dim]
            b = tf.expand_dims(b, axis=0)
            attentional_representations = tf.concat([attentional_representations, b], axis=0)
            idx_var = tf.add(idx_var, 1)
            return attentional_representations, idx_var

        loop_cond = lambda a1, idx: tf.less(idx, inodes_num)
        loop_vars = [attentional_representations, idx_var]
        attentional_representations, idx_var = tf.while_loop(loop_cond, _recurse_q_nodes, loop_vars,
                                                             shape_invariants=[
                                                                 tf.TensorShape(None, 2 * self.hidden_dim),
                                                                 idx_var.get_shape()])
        root_attentional_representation = tf.gather(attentional_representations, tf.substract(q_nodes, 1))
        concated_attentional_rep = tf.concat([context_constituency, root_attentional_representation], axis=0)
        return concated_attentional_rep