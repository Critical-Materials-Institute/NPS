"""
 Simple two-level AMR using TF
"""
import numpy as np
np.set_printoptions(edgeitems=30, linewidth=200, formatter=dict(float=lambda x: "%.3g" % x))
import tensorflow.compat.v1 as tf
from NPS_common.utils import grid_points, digits2int, linear_interp_coeff
from NPS_common.tf_utils import int2digits

class amr_state_variables:
    # def __init__(self, dim, shape_all, shape0, mask1, mask1_where, field0, field1, shared_interface, shared_edge3d, mask_all, field_all, edge_all, type_all=None, refine_threshold=0.001, periodic=True):
    def __init__(self, dim, shape_all, shape0, field_all, type_all=None, refine_threshold=0.001, periodic=True, buffer=0, eval_freq=1):
        # """"
        #   shared_interface: flag of shape [shape0, dim, 2]
        #   shared_edge3d: [shape0, dim, dim, 2, 2]
        # """"
        assert periodic, 'Must be periodic'
        self.dim = dim
        self.shape_all = tuple(shape_all)
        self.n_all = np.prod(shape_all)
        self.ijk2int = tf.convert_to_tensor([shape_all[1],1] if dim==2 else [shape_all[1]*shape_all[2],shape_all[2],1], dtype=tf.int64)
        # edge_all = tf.tensordot(edge_all, ijk2int, 1)
        self.shape0 = tuple(shape0)
        shape0 = np.array(shape0)
        self.n0 = np.prod(shape0)
        shape1 = np.array(shape_all) // np.array(shape0)
        self.shape1 = tuple(shape1)
        self.shape1_plus1 = (-1,) + tuple(np.array(shape1)+1)
        assert np.all(np.array(shape_all) == (np.array(shape0)*shape1)), 'Incompatible shapes'
        self.interpolator1 = tf.constant(linear_interp_coeff(self.shape1).astype(np.float32))
        # mesh0 = np.reshape(np.stack(np.meshgrid(*[np.arange(0,shape_all[d],shape1[d]) for d in range(dim)], indexing='ij'),-1), (-1, dim))
        mesh0_0 = grid_points(shape0-1)
        mesh0 = mesh0_0*shape1
        self.mesh0 = tf.convert_to_tensor(mesh0)
        # indices_per1_corner = np.reshape(np.stack(np.meshgrid(*[[0,N] for N in shape1], indexing='ij'),-1), (1,-1, dim))
        indices_per1_corner = grid_points([1]*dim)*shape1
        self.n_per1_corner = 2**dim
        # self.mesh0_corner = tf.convert_to_tensor(((mesh0[:,None] + indices_per1_corner) % shape_all).reshape((-1,dim)))
        self.mesh0_corner_ = tf.convert_to_tensor(((mesh0[:,None] + indices_per1_corner) % shape_all))
        # indices_per1_whole = np.reshape(np.stack(np.meshgrid(*[np.arange(N+1) for N in shape1], indexing='ij'),-1), (1,-1, dim))
        indices_per1_whole = grid_points(shape1)
        self.indices_per1_whole = tf.constant(indices_per1_whole)
        self.n_per1_whole = indices_per1_whole.shape[-2]
        self.mesh0_whole = tf.convert_to_tensor(((mesh0[:,None] + indices_per1_whole) % shape_all))
        # print(f'debug mesh0 {self.mesh0.shape} mesh0corner {self.mesh0_corner.shape} mesh0_whole {self.mesh0_whole.shape}')
        self.field_all = tf.convert_to_tensor(field_all)
        assert self.field_all.shape[-1] == 1, ValueError(f'Only one field implemented so far, got {self.field_all.shape[-1]}')
        # self.mask1 = mask1
        # self.mask1_where = mask1_where
        # self.field0 = field0
        # self.field1 = field1
        # self.shared_interface = shared_interface
        # self.shared_edge3d = shared_edge3d
        self.periodic = periodic
        self.buffer = buffer
        self.eval_freq = eval_freq
        # self.n1 = mask1_where.shape[0]
        # self.n1_all = np.prod(shape0)
        # self.mask_all = mask_all
        # self.field_all = field_all
        # self.edge_all = edge_all
        type_all = np.zeros(self.shape_all+(1,), dtype=np.int32) if type_all is None else np.reshape(np.array(type_all, dtype=np.int32), self.shape_all+(1,))
        self.type_all = tf.convert_to_tensor(type_all)
        self.refine_threshold = refine_threshold
        # mesh_all = grid_points(np.array(shape_all)-1)# np.reshape(np.stack(np.meshgrid(*[np.arange(N) for N in shape_all], indexing='ij'),-1), (-1, dim))
        mesh_all = grid_points(np.array(shape_all)-1)
        self.mesh_all = tf.convert_to_tensor(mesh_all)
        indices_neighbor = np.stack([np.zeros((dim,dim)),np.eye(dim)], 1) # shape [dim, 2, dim]
        self.edges_all = tf.convert_to_tensor(((mesh_all[:,None,None] + indices_neighbor) % shape_all).reshape((-1,2,dim)))
        self.indices_neighbor = tf.constant(np.eye(dim,dtype=np.int64)[None,...]) # shape [dim, dim]
#        self.indices_neighbor_alldir = tf.constant((grid_points([2]*dim)-1).reshape(1,-1,dim)) # all points made from [-1,0,1]
        indices_neighbor_alldir = (grid_points([2]*dim)-1).reshape(1,-1,dim) # all points made from [-1,0,1]
        self.mesh0_nbcell = tf.constant((mesh0_0[:,None,...] + indices_neighbor_alldir) % self.shape0)
        self.indices_neighbor_level0 = tf.constant(np.diag(shape1)[None,...]) # shape [dim, dim]
        mask_all = np.ones(self.shape_all, dtype=np.int32)
        self.mask_all = tf.convert_to_tensor(mask_all)
        mask_all_0 = np.zeros(shape_all, dtype=np.int32)
        # print('ix ', [np.arange(0,shape_all[d],shape1[d]) for d in range(dim)])
        mask_all_0[np.ix_(*[np.arange(0,shape_all[d],shape1[d]) for d in range(dim)])] = 1
        self.mask_all_0 = tf.constant(mask_all_0)
        self.mesh = tf.where(self.mask_all)
        self.refine_flag = tf.convert_to_tensor(np.full(self.n0, True))
        self.coarse_flag = tf.convert_to_tensor(np.full(self.n0, False))
        self.refine_index = tf.where(self.refine_flag)
        self.coarse_index = tf.where(self.coarse_flag)
        # print(f'debug init mesh {self.mesh.shape} {self.mask_all.shape}')
        # self.get_valid_edges()


    def update_field_from_graph(self, field, mesh=None, update=True):
        mesh = self.mesh if mesh is None else mesh
        # print('right before update field', tf.reduce_max(self.field_all), tf.reduce_min(self.field_all))
        field = tf.scatter_nd(mesh, field, self.shape_all+(1,))
        if update:
            self.field_all = field
        return field
        # print('right after update field', tf.reduce_max(self.field_all), tf.reduce_min(self.field_all))
        # plot_amr(self.mesh.numpy(), field_on_mesh.numpy(), self.edges.numpy())


    # def update_topology(self):
    #     field0 = tf.reshape(tf.gather_nd(self.field_all, self.mesh0_corner), (-1,self.n_per1_corner))
    #     refine_flag = (tf.math.reduce_max(field0,1) - tf.math.reduce_min(field0,1)) > self.refine_threshold
    #     field1 = tf.reshape(tf.gather_nd(self.field_all, self.mesh0_whole), (-1,self.n_per1_whole))
    #     print(f'debug field0 {field0.shape} field1 {field1.shape}')
    #     coarse_flag = (tf.math.reduce_max(field1,1) - tf.math.reduce_min(field1,1)) <= self.refine_threshold
    #     print(f'debug refine flag {refine_flag.shape} {tf.where(refine_flag).shape} coarse_flag {coarse_flag.shape} {tf.where(coarse_flag).shape}')
    #     refine_flag = tf.logical_not(coarse_flag)
    #     # quick return possible but not implemented here because of stupid tf 1.x
    #     valid = tf.where(tf.reshape(refine_flag, self.shape0))
    #     print(f'debug valid {valid.shape}')
    #     valid = tf.reshape((valid[:,None]*self.shape1 + self.indices_per1_whole) % self.shape_all, (-1,dim))
    #     print(f'debug valid {valid.shape} self.indices_per1_whole {self.indices_per1_whole.shape}')
    #     mask_all = tf.tensor_scatter_nd_update(self.mask_all, valid, tf.ones(valid.shape[0], dtype=tf.int32), self.shape_all)
    #     print(f'debug update_topology tensor_scatter_nd_update self.mask_all, valid, tf.ones(valid.shape[0], dtype=tf.int32), self.shape_all {self.mask_all.shape}, {valid.shape}, {tf.ones(valid.shape[0], dtype=tf.int32).shape}  { self.shape_all}')
    #     self.mask_all = tf.cast(tf.cast(mask_all, tf.bool), tf.int32)
    #     self.mesh = tf.where(self.mask_all)
    #     print(f'debug update mesh {self.mesh.shape} {self.mask_all.shape}')

    def interpolate_field(self, field0, refine_idx, refine_in_coarse_index, field_all=None):
        field_all = self.field_all if field_all is None else field_all
        # intp_val = tf.reshape(tf.tensordot(tf.gather_nd(field0, refine_idx), self.interpolator1, 1), (-1,1))
        # print(f'debug refine_in_coarse_index {refine_in_coarse_index.shape} tf.gather_nd(self.field_all, refine_in_coarse_index) {tf.gather_nd(self.field_all, refine_in_coarse_index).shape}')
        intp_val = tf.reshape(tf.tensordot(tf.gather_nd(field0, refine_idx), self.interpolator1, 1), (-1,1))
        # intp_index = tf.reshape(((int2digits(tf.reshape(refine_in_coarse_index,[-1]), self.shape0)*self.shape1)[:,None,:] + self.indices_per1_whole) % self.shape_all, (-1,self.dim))
        intp_index = tf.reshape(tf.gather_nd(self.mesh0_whole, refine_in_coarse_index), (-1,self.dim))
        # print(f'debug refine_in_coarse refine_idx intp_index {refine_in_coarse.shape} { refine_idx.shape} { intp_index.shape} ')
        # intp_val = tf.reshape(tf.tensordot(tf.gather_nd(field0, refine_in_coarse), self.interpolator1, 1), self.shape1_plus1)
        # print(f'debug inptval {intp_val.shape} intpidx {intp_index.shape} all {self.field_all.shape} tf.gather_nd(field0, refine_idx), self.interpolator1 {tf.gather_nd(field0, refine_idx).shape } {self.interpolator1.shape}')
        # print('right before interpolation', tf.reduce_max(self.field_all), tf.reduce_min(self.field_all))
        # plot_amr(self.mesh.numpy(), tf.gather_nd(self.type_all, self.mesh).numpy(), self.edges.numpy())
        return tf.tensor_scatter_nd_update(field_all, intp_index, intp_val)
        # print('right after interpolation', tf.reduce_max(self.field_all), tf.reduce_min(self.field_all))
        # plot_amr(self.mesh.numpy(), tf.gather_nd(self.type_all, self.mesh).numpy(), self.edges.numpy())


    def get_mask(self, refine_index):
        ## refine_index = self.refine_index if refine_index is None else refine_index
        ## valid = tf.where(tf.reshape(self.refine_flag, self.shape0))
        # valid0 = int2digits(refine_index, self.shape0)
        # valid0 = tf.reshape((valid0*self.shape1 + self.indices_per1_whole) % self.shape_all, (-1,self.dim))
        valid0 = tf.reshape(tf.gather_nd(self.mesh0_whole, refine_index), (-1,self.dim))
        mask_all = tf.tensor_scatter_nd_update(self.mask_all_0, valid0, tf.ones(tf.shape(valid0)[0], dtype=tf.int32))
        return mask_all
        # mask_all = tf.scatter_nd_update(valid, tf.ones(valid.shape[0], dtype=tf.int32), self.shape_all)
        # self.mask_all = tf.cast(tf.cast(mask_all, tf.bool), tf.int32)


    def update_topology(self, mesh=None, field_all=None, refine_index=None, coarse_index=None, update=True, index=False):
        # refine/coarse_index in [0, N0)
        mesh = self.mesh if mesh is None else mesh
        field_all = self.field_all if field_all is None else field_all
        refine_index = self.refine_index if refine_index is None else refine_index
        coarse_index = self.coarse_index if coarse_index is None else coarse_index
        nrefine_old = tf.size(refine_index)
        ncoarse_old = tf.size(coarse_index)
        # which of fine mesh0 to coarsen?
        # refine_index = tf.where(self.refine_flag)
        field1 = tf.reshape(tf.gather_nd(field_all, tf.reshape(tf.gather_nd(self.mesh0_whole, refine_index),(-1,self.dim))), (-1,self.n_per1_whole))
        coarse_in_refine = (tf.math.reduce_max(field1,1) - tf.math.reduce_min(field1,1)) <= self.refine_threshold
        refine_in_refine_index = tf.gather(refine_index, tf.where(tf.logical_not(coarse_in_refine)))

        # coarse_flag = tf.tensor_scatter_nd_update(self.coarse_fla)

        # which of coarse mesh0 to refine, i.e. interpolate?
        # coarse_index = tf.where(self.coarse_flag)
        field0 = tf.reshape(tf.gather_nd(field_all, tf.reshape(tf.gather_nd(self.mesh0_corner_, coarse_index),(-1,self.dim))), (-1,self.n_per1_corner))
        # print(f'debug field0 {field0.shape} {field0} field1 {field1.shape} {field1}')
        refine_in_coarse = (tf.math.reduce_max(field0,1) - tf.math.reduce_min(field0,1)) >  self.refine_threshold
        refine_idx =  tf.where(refine_in_coarse)
        refine_in_coarse_index = tf.gather(coarse_index, refine_idx)
        # print(f'debug refine_index before update {refine_index}')
        # print(f'debug  coarse_in_refine {coarse_in_refine} refine_in_refine_index {refine_in_refine_index} refine_in_coarse {refine_in_coarse} refine_idx {refine_idx} refine_in_coarse_index {refine_in_coarse_index}')

        # synthesize new refine_flag
        # print(f'debug refine_in_refine_index, refine_in_coarse_index {refine_in_refine_index.shape} {refine_in_refine_index}, {refine_in_coarse_index.shape} {refine_in_coarse_index}')
        refine_index = tf.reshape(tf.concat([refine_in_refine_index, refine_in_coarse_index], 0), (-1,1))
        # print(f'debug refine_index after  update {refine_index}')
        # take care of buffer, if any
        if self.buffer == 0:
            refine_flag = tf.scatter_nd(refine_index, tf.fill([tf.size(refine_index)], True), [self.n0])
        elif self.buffer == 1:
#            flagged = int2digits(refine_index, self.shape0)
#            flagged = tf.reshape(flagged + self.indices_neighbor_alldir, (-1, self.dim)) % self.shape0
#            refine_index = (tf.unique(digits2int(flagged, self.shape0))[0])[:,None]
            flagged = tf.reshape(tf.gather_nd(self.mesh0_nbcell, refine_index), (-1, self.dim))
            refine_index = digits2int(flagged, self.shape0)[:,None]
            refine_flag = tf.scatter_nd(refine_index, tf.fill([tf.size(refine_index)], True), [self.n0])
            refine_index = tf.where(refine_flag)
            refine_idx = tf.where(tf.gather_nd(refine_flag, coarse_index))
            refine_in_coarse_index = tf.gather(coarse_index, refine_idx)
        else:
            raise ValueError(f'ERROR: buffer must be 0 or 1')

        # print(f'debug before interpolation refine_index {refine_index} refine_flag {refine_flag} refine_idx {refine_idx} field0 {field0} refine_in_coarse_index {refine_in_coarse_index}')
        # print(f' field_all {field_all[...,0]}')
        field_all = self.interpolate_field(field0, refine_idx, refine_in_coarse_index, field_all)
        # print(f' field_all after {field_all[...,0]}')

        # other book-keeping
        coarse_flag = tf.logical_not(refine_flag)
        coarse_index = tf.where(coarse_flag)
        # print(f'debug field0 {field0.shape} field1 {field1.shape}')
        # print(f'debug refine flag {self.refine_index.shape} coarse_flag {self.coarse_index.shape}')

        # refine_flag = tf.logical_not(coarse_flag)
        # quick return possible but not implemented here because of stupid tf 1.x
        mask_all = self.get_mask(refine_index)
        mesh = tf.where(mask_all)
        # print(f'debug update mesh {self.mesh.shape} {self.mask_all.shape}')
        # print(f'Updates: coarsened/fine= {nrefine_old-tf.size(refine_in_refine_index)}/{nrefine_old}, refined/coarse={tf.size(refine_in_coarse_index)}/{ncoarse_old}, NEW fine= {tf.size(refine_index)} mesh= {mesh.shape[0]}')
        # self.get_valid_edges()
        if update:
            self.mask_all = mask_all
            self.mesh = mesh
            self.field_all = field_all
            self.refine_index = refine_index
            self.refine_flag = refine_flag
            self.coarse_flag = coarse_flag
            self.coarse_index = coarse_index
        if index:
            return mask_all, mesh, field_all, refine_index, coarse_index
        else:
            return mask_all, mesh, field_all


    def get_valid_edges(self, mesh=None, mask_all=None, update=True):
        # fine edges
        mesh = self.mesh if mesh is None else mesh
        mask_all = self.mask_all if mask_all is None else mask_all
        edges_A = tf.reshape(tf.tile(mesh, [1,self.dim]), (-1, self.dim))
        edges_B = tf.reshape((mesh[:,None] + self.indices_neighbor) % self.shape_all, (-1, self.dim))
        valid = tf.where(tf.gather_nd(mask_all, edges_B))
        # print(f'debug a {edges_A.shape} b {edges_B.shape} mask_all {mask_all.shape} valid {valid.shape}')
        edges_lv1 = tf.tensordot(tf.gather_nd(tf.stack([edges_A, edges_B],1), valid), self.ijk2int, 1)
        # print(f'debug level1 (fine) edges', {edges_lv1.__class__})
        # coarse edges
        edges_A = tf.reshape(tf.tile(self.mesh0, [1,self.dim]), (-1, self.dim))
        edges_B = tf.reshape((self.mesh0[:,None] + self.indices_neighbor_level0) % self.shape_all, (-1, self.dim))
        edges_Aplus1 = tf.reshape((self.mesh0[:,None] + self.indices_neighbor) % self.shape_all, (-1, self.dim))
        valid = tf.where(tf.equal(tf.gather_nd(mask_all, edges_Aplus1), 0)) # add missing coarse edges only
        # print(f'debug valid {valid.shape} edges_A {edges_A.shape} edges_B {edges_B.shape} self.ijk2int {self.ijk2int.shape}')
        # print(f'tf ver {tf.__version__}')
        # print(f'debug ab {tf.stack([edges_A, edges_B],1).shape}')
        # print(f'debug edgeA {edges_A} edgeB {edges_B} self.indices_neighbor_level0 {self.indices_neighbor_level0} sum {self.mesh0[:,None] + self.indices_neighbor_level0} within {(self.mesh0[:,None] + self.indices_neighbor_level0) % self.shape_all}')
        edges_lv0 = tf.tensordot(tf.gather_nd(tf.stack([edges_A, edges_B],1), valid), self.ijk2int, 1)
        # print(f'debug  edges_lv1 {edges_lv1.__class__} edges_lv0 {edges_lv0.shape}')
        edges = tf.concat([edges_lv1, edges_lv0], 0)
        # print(f'debug edges {edges.shape}')
        # now map index in whole region to index in returned mesh
        index_final = tf.cumsum(tf.reshape(mask_all, [-1]))-1
        edges = tf.reshape(tf.gather(index_final, tf.reshape(edges,[-1])), (-1,2))
        # print(f'debug edge min {tf.reduce_min(edges)} max {tf.reduce_max(edges)} mesh {tf.shape(mesh)}')
        if update:
            self.edges = edges
        return edges
        

    def to_graph(self, mesh=None, mask_all=None, field_all=None, field_tgt=None, scale=1.0, type_all=None, update=True):
        """
        output graph depending on current mask_all and mesh=where(mask_all)
        returns:
          mesh positions [N, dim, dtype=float32]
          mesh type      [N, dtype=int32]
          field value    [N, ?, dtype=float32]
          target value   [N, ?, dtype=float32] 
          edges          [N_edges, 2, dtype=int64]
        """
        # print(f'debug mesh {self.mesh.shape}')
        mesh = self.mesh if mesh is None else mesh
        mask_all = self.mask_all if mask_all is None else mask_all
        type_all = self.type_all if type_all is None else type_all
        field_all = self.field_all if field_all is None else field_all
        # print(f'debug TO_GRAPH field {tf.gather_nd(field_all, mesh)} field_all {field_all[...,0]} mesh {mesh.shape}')
        # print(f'debug field tgt {field_tgt}')
        return tf.cast(mesh, tf.float32)*scale, \
          tf.gather_nd(type_all, mesh), \
          tf.gather_nd(field_all, mesh), \
          tf.constant(0.) if field_tgt is None else tf.gather_nd(field_tgt, mesh), \
          self.get_valid_edges(mesh, mask_all, update=update)


    def adapt(self, field_on_mesh):
        self.update_field_from_GNN(field_on_mesh)
        self.update_topology()
        return self.to_graph()


    def remesh_input(self, mesh, field_inp, field_tgt, refine_index=None, coarse_index=None, scale=1.0, input_dense=True, index=False):
        mesh = tf.cast(mesh, tf.int32)
        # print(f'debug in remesh_input')
        if input_dense:
            # simply reshape
            field_all = tf.reshape(field_inp, self.shape_all+(1,))
            field_tgt = tf.reshape(field_tgt, self.shape_all+(1,)) if field_tgt is not None else None
        else:
            field_all = self.update_field_from_graph(field_inp, mesh=mesh, update=False)
            field_tgt = tf.transpose(tf.reshape(field_tgt, self.shape_all))[...,None] if field_tgt is not None else None # if field_tgt is not None else field_tgt
        # ## field_tgt is always dense!!!
        # print(f'debug in remesh_input field_tgt {field_tgt}')
        # ## WARNING: temporary fix
        # print(f'debug in remesh_input field_all {field_all} tgt {field_tgt} mesh {mesh}')
        if index:
            mask_all, mesh, field_all, refine_index, coarse_index = self.update_topology(mesh, field_all, refine_index=refine_index, coarse_index=coarse_index, update=False, index=index)
            return self.to_graph(mesh, mask_all, field_all, field_tgt, scale=scale, update=False) + (refine_index, coarse_index)
        else:
            mask_all, mesh, field_all = self.update_topology(mesh, field_all, update=False, index=index)
            # print(f'debug in remesh_input mesh {mesh} field_all {field_all}')
            return self.to_graph(mesh, mask_all, field_all, field_tgt, scale=scale, update=False)


    #### @classmethod
    # def from_dense(cls, dim, shape_dense, shape0, field, mask=None, periodic=True):
    #     """
    #     dense: field e.g. of shape [64, 64, 1]
    #     """
    #     assert periodic, 'Must be periodic'
    #     shape1 = np.array(shape_dense) // np.array(shape0)
    #     shape0_tuple = tuple(np.array(shape0).tolist())
    #     shape1_tuple = tuple(np.array(shape1).tolist())
    #     print(f'debug {str(shape_dense)} {str(shape0)} {str(shape1)}')
    #     shape_block = tf.reshape(tf.stack([shape0,shape1],1),-1).numpy().tolist()
    #     if mask is None:
    #         mask0 = tf.ones(shape0, dtype=tf.int8)
    #     else:
    #         mask0 = tf.cast(tf.reshape(mask, shape_block), dtype=tf.int8)
    #         # print(f'mask 1 {mask1.shape} ')
    #         mask0 = tf.math.reduce_max(mask0, axis=list(range(1,2*dim,2)))
    #         # print(f'mask 1 {mask1.shape} ')
    #     # shared_interface = tf.sparse.reorder(tf.sparse.SparseTensor([[5],[1]], 1, tf.concat(shape0, [dim, 2]),dtype=tf.int8))
    #     mask1_where = tf.where(mask1)
    #     n1 = mask1_where.shape[0]
    #     print(f'Level 1: masked {mask1_where.shape} all level 1{mask1.shape}')

    #     indices_per_1 = tf.convert_to_tensor(np.reshape(np.stack(np.meshgrid(*[np.arange(N+1) for N in shape1], indexing='ij'),-1), (1,-1, dim)))
    #     indices_all = tf.convert_to_tensor(np.reshape(np.stack(np.meshgrid(*[np.arange(N) for N in shape_dense], indexing='ij'),-1), (-1, dim)))
    #     print(f'debug indices_per_1 {indices_per_1.shape} indices_all {indices_all.shape}')

    #     indices = tf.math.floormod(tf.reshape(mask1_where[:,tf.newaxis,:]+indices_per_1[tf.newaxis,:,:], (-1,dim)), tf.convert_to_tensor(shape_dense[None,:]))
    #     print(f'debug indices {indices.shape}')
    #     mask_all = tf.scatter_nd(indices, tf.ones([indices.shape[0]], dtype=tf.int8), shape_dense)
    #     print(f'debug mask_all {mask_all.shape}')

    #     # edge_all = [ for i, idx in enumerate(indices_all)  tf.eye(dim,dtype=tf.int64)]
    #     edge_all = tf.math.floormod(indices_all[:,tf.newaxis,:] + tf.eye(dim,dtype=tf.int64)[tf.newaxis,:,:], tf.convert_to_tensor(shape_dense[None,None,:]))
    #     ijk2int = tf.convert_to_tensor([shape_dense[1],1] if dim==2 else [shape_dense[1]*shape_dense[2],shape_dense[2],1], dtype=tf.int64)
    #     edge_all = tf.tensordot(edge_all, ijk2int, 1)
    #     print(f'debug edge_all {edge_all.shape}')

    #     field_padded = wrap_pad_right(field, 1, dim=dim)
    #     field_all_padded = [field_padded[r[0]*shape1[0]:(r[0]+1)*shape1[0]+1, r[1]*shape1[1]:(r[1]+1)*shape1[1]+1] if dim==2 else 
    #               field_padded[r[0]*shape1[0]:(r[0]+1)*shape1[0]+1, r[1]*shape1[1]:(r[1]+1)*shape1[1]+1, r[2]*shape1[2]:(r[2]+1)*shape1[2]+1] for r in tf.where(tf.ones(shape0))]
    #     field_all = field
    #     print(f'debug field_all_padded {len(field_all_padded)} {field_all_padded[0].shape} field_all {tf.shape(field_all)}')

    #     # print(mask1_where.shape, tf.ones([mask1_where.shape[0], dim,2],dtype=tf.int8).shape, )
    #     vals = tf.ones([n1, dim],dtype=tf.int8)
    #     shared_interface = tf.scatter_nd(mask1_where, tf.stack([tf.zeros_like(vals), vals], -1), shape0_tuple+(dim, 2))
    #     # shared_interface = tf.zeros(shape0_list+[dim, 2], dtype=tf.int8)
    #     dirs = tf.eye(dim, dtype=tf.int64)[tf.newaxis,...]
    #     # for d in range(dim):
    #     #     shared_interface = tf.tensor_scatter_nd_add(shared_interface, tf.concat([tf.math.floormod(mask1_where+dirs[:,d], shape0), tf.fill([n1] d), tf.ones([n1, 0],dtype=tf.int8)],-1), vals[...,0])
    #     # print(shared_interface.shape, shared_interface)
    #     shared_edge3d = None
    #     if dim == 3:
    #         vals = tf.ones([n1, dim],dtype=tf.int8)
    #         shared_edge3d = tf.scatter_nd(mask1_where, tf.stack([tf.zeros_like(vals), vals], -1), shape0_tuple+(dim, 2))
    #         shared_edge3d = tf.scatter_nd(mask1_where, tf.ones([mask1_where.shape[0], dim,2,2],dtype=tf.int8), shape0_tuple+(dim,2,2))
    #     print(f'debug interface {shared_interface.shape} edge3d {shared_edge3d}')
    #     field0 = field[::shape1[0], ::shape1[1]] if dim==2 else field[::shape1[0], ::shape1[1], ::shape1[2]]
    #     field_padded = wrap_pad_right(field, 1, dim=dim)
    #     field1 = [field_padded[r[0]*shape1[0]:(r[0]+1)*shape1[0]+1, r[1]*shape1[1]:(r[1]+1)*shape1[1]+1] if dim==2 else 
    #               field_padded[r[0]*shape1[0]:(r[0]+1)*shape1[0]+1, r[1]*shape1[1]:(r[1]+1)*shape1[1]+1, r[2]*shape1[2]:(r[2]+1)*shape1[2]+1] for r in mask1_where]
    #     field1 = tf.stack(field1)
    #     # field1 = tf.reshape(field, shape_block+[field.shape[-1]])
    #     # if dim == 2:
    #     #     field1 = tf.transpose(field1, [0,2,1,3,4])
    #     # elif dim == 3:
    #     #     field1 = tf.transpose(field1, [0,2,4,1,3,5,6])
    #     print(f'field 1 {field1.shape} field 0 {field0.shape}')
    #     # field1 = tf.
    #     return cls(dim, shape_dense, shape0, shape1, mask1, mask1_where, field0, field1, shared_interface, shared_edge3d, mask_all, field_all, edge_all, periodic=periodic)


# def amr(shape0, shape1, mask1_prev, mask, field0, field1):
#     print('fcoarse grid {shape0} fine grid {shape1}')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import tri as mtri
    from matplotlib import collections  as mc
    from matplotlib.backends.backend_pdf import PdfPages

    def my_show():
        try:
            pdf.savefig()
        except:
            plt.show()

    def plot_amr(mesh, field, edges, colorbar=False):
        fig1, ax1 = plt.subplots(figsize=(7,7))
        triang = mtri.Triangulation(mesh[:,0], mesh[:,1], None)
        ax1.set_aspect(1)
        tpc = ax1.tripcolor(triang, field.ravel(), vmin=0, vmax=1)
        if colorbar:
            fig1.colorbar(tpc)
        lines = mesh[edges]
        pbc_flag = np.where(np.linalg.norm(lines[:,0]-lines[:,1],axis=1)< (N1+1))
        lines = lines[pbc_flag]
        ax1.add_collection(mc.LineCollection(lines, colors='r', linewidths=0.7)) #, colors=np.random.rand([len(edges),3]), linewidths=1.1))
        my_show()

    import time
    from my_libs.tf1_utils import compatible_run_tf

    if True:
        print('saving to test.pdf')
        pdf = PdfPages('test.pdf')
    else:
        pdf = None

    print('Testing')
    dim=2
    N=64
    N1=2
    N0=N//N1
    mesh_pos = grid_points([N-1]*dim)
    phi = 1-tf.exp(-(tf.norm(mesh_pos - tf.convert_to_tensor([[N/2,N/3]]),axis=1, ord=1) - 23)**2/4)
    # print(phi)
    phi = tf.reshape(phi,[N]*dim + [1])
    phis = tf.reshape([1-tf.exp(-(tf.norm(mesh_pos - tf.convert_to_tensor([[N/2+1.8*t,N/3+0.9*t]]),axis=1, ord=2) - 23)**2/4) for t in range(10)], [-1]+[N]*dim + [1])
    # phis = tf.reshape([1-tf.exp(-(tf.norm(mesh_pos - tf.convert_to_tensor([[N/2+0.8*t,N/3+0.3*t]]),axis=1, ord=2) - 0.02)**2/0.5) for t in range(10)], [-1]+[N]*dim + [1])

    print('\n********************')
    print('Testing manually')
    state = amr_state_variables(dim, np.full(dim, N), np.full(dim, N0), phi, refine_threshold=3e-4, buffer=0)
    mesh, typ, field, target, edges = compatible_run_tf(state.to_graph())
    plot_amr(mesh, field, edges)

    state.update_topology()
    mesh, typ, field, target, edges = compatible_run_tf(state.to_graph())
    plot_amr(mesh, field, edges)

    print('no field update, so nothing should change')
    state.update_topology()
    mesh, typ, field, target, edges = compatible_run_tf(state.to_graph())
    plot_amr(mesh, field, edges)

    print('\n********************')
    for buffer in (0, 1):
        print('Testing TRAINING job remesh_input buffer=', buffer)
        state = amr_state_variables(dim, np.full(dim, N), np.full(dim, N0), phis[0], refine_threshold=3e-4, buffer=buffer)
        for i in (3,6,9):
            mesh, typ, field, target, edges = compatible_run_tf(state.remesh_input(state.mesh, phis[i], phis[i], input_dense=True))
            plot_amr(mesh, field, edges)
            plot_amr(mesh, target, edges)

    print('\n********************')
    for buffer in (0, 1):
        print('Testing EVAL job remesh_input buffer=', buffer)
        state = amr_state_variables(dim, np.full(dim, N), np.full(dim, N0), phis[0], refine_threshold=3e-4, buffer=buffer)
        mesh = state.mesh # all mesh
        field = tf.reshape(phis[0], (-1,1))
        refine_index = state.refine_index; coarse_index = state.coarse_index
        for i in (3,6,9):
            mesh, typ, field, target, edges, refine_index, coarse_index = compatible_run_tf(state.remesh_input(mesh, field, phis[i],
              refine_index=refine_index, coarse_index=coarse_index, input_dense=False, index=True))
            plot_amr(mesh, field, edges)
            plot_amr(mesh, target, edges)
            field = tf.gather_nd(phis[i], tf.cast(mesh,tf.int32))

    for buffer in (0, 1):
        print('\n********************')
        print('Testing adapting to new values in a loop by updating interval variables implicitly buffer=', buffer)
        state = amr_state_variables(dim, np.full(dim, N), np.full(dim, N0), phis[0], refine_threshold=3e-4, buffer=buffer)
        start = time.time()
        state.update_topology()
        end = time.time()
        print(' first time to update tpology', end - start)
        start = time.time()
        mesh, typ, field, target, edges = compatible_run_tf(state.to_graph())
        end = time.time()
        print(' first time to get results', end - start)
        for i in range(10):
            print(f'  Step {i}')
            start = time.time()
            state.update_field_from_graph(tf.gather_nd(phis[i],tf.cast(mesh,tf.int32)))
            end = time.time()
            print(' time to update field', end - start)
            start = time.time()
            state.update_topology()
            end = time.time()
            print(' time to update', end - start)
            start = time.time()
            mesh, typ, field, target, edges = compatible_run_tf(state.to_graph())
            end = time.time()
            print(' time to run', end - start)
            plot_amr(mesh, field, edges)

    if pdf is not None:
        pdf.close()

# print(gnn_vars)
# mesh, typ, field, target, edges = gnn_vars

# mesh, typ, field, edges = list(map(lambda x: x.numpy(), (mesh, typ, field, edges)))
# print(mesh.shape, field.shape)




# plt.matshow(tf.reshape(phi,(N,N)), cmap='gray'); plt.show()
# ax1.tripcolor(x=mesh.numpy()[:,0], y=mesh.numpy()[:,1], C=field.numpy().ravel())#, vmin=vmin[0], vmax=vmax[0])
# print(gnn_vars[-1][:,0]==gnn_vars[-1][:,1])

# # print(digits2int(tf.constant([[1,0]]), tf.constant([64, 32])))
# # print(int2digits(digits2int(tf.constant([[1,0]]), tf.constant([64, 32])), tf.constant([64, 32])))

# def amr_tf(mesh_pos, cells, velocity, target_velocity, lattice, mask, fill_val=1.0, padding=2, periodic=True):
# # mask is a mask marking significant nodes
#     dim = mesh_pos.shape[-1]
#     shape = tf.cast(tf.linalg.diag_part(lattice), tf.int32)
#     shape1 = tf.expand_dims(shape, 0)
#     # print(shape, shape1)
#     # quick return if possible
#     n_tot = tf.reduce_prod(shape)
#     if n_tot == mesh_pos.shape[0] and tf.reduce_all(mask):
#         print("DEBUG all points remaining, so returning quickly")
#         return mesh_pos, cells, tf.zeros(n_tot), velocity, target_velocity

#     neighbors = tf.reshape(tf.stack(tf.meshgrid(*([list(range(-padding,padding+1))]*dim), indexing='ij'),-1), (-1, dim))
#     # print('nb', neighbors.shape)
#     all_digits = tf.cast(mesh_pos, tf.int32)
#     all_int = digits2int(all_digits, shape)

#     mask_index = tf.where(mask)
#     new_digits = tf.expand_dims(tf.gather_nd(all_digits, mask_index),1) + tf.expand_dims(neighbors,0)
#     print('new_digits shape', new_digits.shape)
#     new_digits = tf.reshape(new_digits, (-1,dim))
#     print('new_digits shape', new_digits.shape)
#     if periodic:
#         new_digits = tf.math.floormod(new_digits, shape1)
#     else:
#         new_digits = tf.gather_nd(new_digits, tf.where(tf.reduce_all(new_digits>=0, axis=1) & tf.reduce_all(new_digits<shape1, axis=1)))
#     print('new_digits shape', new_digits.shape)
#     new_int = tf.unique(digits2int(new_digits, shape))[0]
#     new_digits = int2digits(new_int, shape)
#     new_ntot = new_int.shape[0]
#     keep_int = tf.sparse.to_dense(tf.sets.intersection(tf.expand_dims(new_int,0), tf.expand_dims(all_int,0)))
#     added_int = tf.sparse.to_dense(tf.sets.difference(tf.expand_dims(new_int,0), tf.expand_dims(all_int,0)))
#     print(f'DEBUG new nodes: tot {new_ntot} kept {tf.size(keep_int)} added {tf.size(added_int)}')


#     all_digits = tf.cast(mesh_pos, tf.int64)
#     shape = tf.cast(tf.linalg.diag_part(lattice), tf.int64)
#     # mask_sp = tf.sparse.SparseTensor(all_digits, tf.cast(mask, tf.int8), shape)
#     mask_sp = tf.sparse.SparseTensor(tf.cast(new_digits, tf.int64), tf.ones(new_digits.shape[0], dtype=tf.int8), shape)
#     print(f'mask_sp {mask_sp}')
#     # print(f'mask_sp {tf.sparse.add(mask_sp, mask_sp)}')
#     v_sp = tf.sparse.SparseTensor(all_digits, velocity, shape)
#     vtarget_sp = tf.sparse.SparseTensor(all_digits, target_velocity, shape)
#     neighbors = tf.reshape(tf.stack(tf.meshgrid(*([list(range(-padding,padding+1))]*dim), indexing='ij'),-1), (-1, dim))
#     neighbors = tf.gather_nd(neighbors, tf.where(tf.reduce_any(neighbors !=0,axis=1)))
#     neighbors = tf.cast(neighbors, tf.int8)
#     # print('nb', neighbors)
#     # for nb in neighbors:
#     #     mask_sp = tf.sparse.add(mask_sp, tf.sparse.SparseTensor(mask_sp.indices, mask_sp.values, mask_sp.shape))


#     new_v = tf.fill([new_ntot], fill_val)
#     new_target_v = tf.fill([new_ntot], fill_val)
#     return tf.cast(new_digits, tf.float32), new_cells, tf.zeros(new_ntot), new_v, new_target_v

#     all_pos = tf.expand_dims(tf.cast(mesh_pos,tf.int32),1) + tf.expand_dims(neighbors,0)
#     all_pos = tf.reshape(all_pos, (-1,dim))
#     all_pos = tf.math.floormod(all_pos, tf.expand_dims(shape,0))
#     # remove duplicate points
#     all_pos = tf.bitcast(tf.unique(tf.bitcast(all_pos, tf.int64))[0], tf.int32)

#     print('all pos', all_pos.shape, all_pos)
#     node_index_from_original = tf.scatter_nd(indices, updates, shape)
#     new_pos = tf.gather_nd(mesh_pos, node_index)
#     print('kept nodes', node_index.shape )

#     index_int = pos_int[0] * shape[1] + pos_int[1]
#     cells = []
#     nbs = tf.eye(dim, dtype=tf.int32)
#     for i in range(dim):
#         nb = new_pos + nbs[i:i+1]
#         nb[:,i] = tf.math.floormod(nb[:,i], shape[i])
#         index_nb = nb[0] * shape[1] + nb[1]
#         # nb_ok = tf.
    
#     return cells, new_pos, tf.zeros(tf.size(node_index)), tf.gather_nd(velocity, node_index), tf.gather_nd(target_velocity, node_index)
#      # mesh_pos[node_index,:] #, tf.zeros(tf.size(node_index)), velocity[node_index], target_velocity[node_index]


#     mapping = np.zeros(shape)
#     mapping.fill(-1)
#     if dim == 2:
#         #velocity of each grid point
#         for i, node in enumerate(mesh_pos):
#             x, y = np.round(node).astype(np.int32)
#             v[x + y * shape[0], 0] = velocity[i, 0]
#         velocity = v
#         nodes, velocity_amr, target_velocity_amr, cells = [], [], [], []
#         for i in range(np.prod(mapping.shape)):
#             x, y = i % shape[0], i // shape[0]
#             node = (x, y)
#             #find neighbers of a node
#             neighbors = [[
#                 (x + xp) % shape[0], (y + yp) % shape[1]
#             ] for xp, yp in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1),
#                              (-1, -1), (1, -1))]
#             #remove the node if necessary
#             if min([velocity[xp + yp * shape[0], 0]
#                     for xp, yp in neighbors]) > 0.99:
#                 continue
#             mapping[x, y] = len(nodes)
#             nodes.append(node)
#             #build vertical and horizontal edges
#             for xp, yp in neighbors[:4]:
#                 if mapping[xp, yp] >= 0:
#                     cells.append([mapping[xp, yp], mapping[x, y]])
#             velocity_amr.append(velocity[i])
#             target_velocity_amr.append(target_velocity[i])
#     mesh_pos = np.stack(nodes)
#     velocity = np.stack(velocity_amr)
#     target_velocity = np.stack(target_velocity_amr)
#     node_type = np.zeros((len(mesh_pos), 1), dtype=np.int32)
#     cells = np.array(cells, dtype=np.int32)
#     return cells, mesh_pos, node_type, velocity, target_velocity




# dim=2
# N=64
# lattice= tf.cast(N*tf.eye(dim), tf.float32)
# print(lattice)

# grid= tf.cast(tf.range(N), tf.float32)
# mesh_pos = tf.reshape(tf.stack(tf.meshgrid(*([grid]*dim), indexing='ij'),-1), (-1, dim))
# print(mesh_pos)

# phi = 1-tf.exp(-(tf.norm(mesh_pos - tf.convert_to_tensor([[N/2,N/3]]),axis=1, ord=1) - 23)**2/4)
# print(phi)
# plt.matshow(tf.reshape(phi,(N,N)), cmap='gray'); plt.show()
# velocity = phi
# target_velocity = phi
# cells=None

# # test early return
# mesh_amr, cells_amr, node_amr, v_amr, target_v_amr = amr_tf(mesh_pos, cells, velocity, target_velocity, lattice, tf.abs(1-velocity)>-0.01)
# print(mesh_amr.shape, cells_amr, node_amr.shape, v_amr.shape, target_v_amr.shape, cells)
# # test whole run
# mesh_amr, cells_amr, node_amr, v_amr, target_v_amr = amr_tf(mesh_pos, cells, velocity, target_velocity, lattice, tf.abs(1-velocity)> 0.01, periodic=False)
# print(mesh_amr.shape, cells_amr, node_amr.shape, v_amr.shape, target_v_amr.shape, cells)
# plt.matshow(tf.reshape(phi,(N,N)), cmap='gray'); plt.show()

