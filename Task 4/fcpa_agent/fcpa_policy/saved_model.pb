??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8Ӭ
?
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namelayer_normalization/gamma
?
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:@*
dtype0
?
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namelayer_normalization/beta
?
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:@*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@@*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:@*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:@*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?#@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?#@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
?
skip_dense/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@**
shared_nameskip_dense/dense_1/kernel
?
-skip_dense/dense_1/kernel/Read/ReadVariableOpReadVariableOpskip_dense/dense_1/kernel*
_output_shapes

:@@*
dtype0
?
skip_dense/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameskip_dense/dense_1/bias

+skip_dense/dense_1/bias/Read/ReadVariableOpReadVariableOpskip_dense/dense_1/bias*
_output_shapes
:@*
dtype0
?
skip_dense_1/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*,
shared_nameskip_dense_1/dense_2/kernel
?
/skip_dense_1/dense_2/kernel/Read/ReadVariableOpReadVariableOpskip_dense_1/dense_2/kernel*
_output_shapes

:@@*
dtype0
?
skip_dense_1/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameskip_dense_1/dense_2/bias
?
-skip_dense_1/dense_2/bias/Read/ReadVariableOpReadVariableOpskip_dense_1/dense_2/bias*
_output_shapes
:@*
dtype0

NoOpNoOp
?&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?%
value?%B?% B?%
?

activation
softmax

hidden
normalization
	lastlayer
	out_layer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api

0
1
2
q
axis
	gamma
beta
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
V
*0
+1
,2
-3
.4
/5
6
7
8
9
$10
%11
V
*0
+1
,2
-3
.4
/5
6
7
8
9
$10
%11
 
?
0layer_regularization_losses
	variables
trainable_variables
1layer_metrics
2metrics
	regularization_losses

3layers
4non_trainable_variables
 
 
 
 
?
5layer_regularization_losses
	variables
trainable_variables
6layer_metrics
7metrics
regularization_losses

8layers
9non_trainable_variables
 
 
 
?
:layer_regularization_losses
	variables
trainable_variables
;layer_metrics
<metrics
regularization_losses

=layers
>non_trainable_variables
h

*kernel
+bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
^

Chidden
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
^

Hhidden
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
 
][
VARIABLE_VALUElayer_normalization/gamma.normalization/gamma/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElayer_normalization/beta-normalization/beta/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Mlayer_regularization_losses
	variables
trainable_variables
Nlayer_metrics
Ometrics
regularization_losses

Players
Qnon_trainable_variables
OM
VARIABLE_VALUEdense_3/kernel+lastlayer/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_3/bias)lastlayer/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Rlayer_regularization_losses
 	variables
!trainable_variables
Slayer_metrics
Tmetrics
"regularization_losses

Ulayers
Vnon_trainable_variables
OM
VARIABLE_VALUEdense_4/kernel+out_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_4/bias)out_layer/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
?
Wlayer_regularization_losses
&	variables
'trainable_variables
Xlayer_metrics
Ymetrics
(regularization_losses

Zlayers
[non_trainable_variables
HF
VARIABLE_VALUEdense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEskip_dense/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEskip_dense/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEskip_dense_1/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEskip_dense_1/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 

*0
+1

*0
+1
 
?
\layer_regularization_losses
?	variables
@trainable_variables
]layer_metrics
^metrics
Aregularization_losses

_layers
`non_trainable_variables
h

,kernel
-bias
a	variables
btrainable_variables
cregularization_losses
d	keras_api

,0
-1

,0
-1
 
?
elayer_regularization_losses
D	variables
Etrainable_variables
flayer_metrics
gmetrics
Fregularization_losses

hlayers
inon_trainable_variables
h

.kernel
/bias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api

.0
/1

.0
/1
 
?
nlayer_regularization_losses
I	variables
Jtrainable_variables
olayer_metrics
pmetrics
Kregularization_losses

qlayers
rnon_trainable_variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

,0
-1

,0
-1
 
?
slayer_regularization_losses
a	variables
btrainable_variables
tlayer_metrics
umetrics
cregularization_losses

vlayers
wnon_trainable_variables
 
 
 

C0
 

.0
/1

.0
/1
 
?
xlayer_regularization_losses
j	variables
ktrainable_variables
ylayer_metrics
zmetrics
lregularization_losses

{layers
|non_trainable_variables
 
 
 

H0
 
 
 
 
 
 
 
 
 
 
 
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????#*
dtype0*
shape:??????????#
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2dense/kernel
dense/biasskip_dense/dense_1/kernelskip_dense/dense_1/biasskip_dense_1/dense_2/kernelskip_dense_1/dense_2/biaslayer_normalization/gammalayer_normalization/betadense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_34844046
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp-skip_dense/dense_1/kernel/Read/ReadVariableOp+skip_dense/dense_1/bias/Read/ReadVariableOp/skip_dense_1/dense_2/kernel/Read/ReadVariableOp-skip_dense_1/dense_2/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_34844274
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betadense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense/kernel
dense/biasskip_dense/dense_1/kernelskip_dense/dense_1/biasskip_dense_1/dense_2/kernelskip_dense_1/dense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_34844320??
?
a
E__inference_softmax_layer_call_and_return_conditional_losses_34844061

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_3_layer_call_and_return_conditional_losses_34844127

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
6__inference_layer_normalization_layer_call_fn_34844117

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_348438962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
E__inference_dense_4_layer_call_and_return_conditional_losses_34843949

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
C__inference_dense_layer_call_and_return_conditional_losses_34844165

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?#@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????#::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????#
 
_user_specified_nameinputs
?
a
E__inference_softmax_layer_call_and_return_conditional_losses_34843974

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?k
?
__inference_call_1816189

inputs
inputs_1(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource5
1skip_dense_dense_1_matmul_readvariableop_resource6
2skip_dense_dense_1_biasadd_readvariableop_resource7
3skip_dense_1_dense_2_matmul_readvariableop_resource8
4skip_dense_1_dense_2_biasadd_readvariableop_resource4
0layer_normalization_cast_readvariableop_resource6
2layer_normalization_cast_1_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?'layer_normalization/Cast/ReadVariableOp?)layer_normalization/Cast_1/ReadVariableOp?)skip_dense/dense_1/BiasAdd/ReadVariableOp?(skip_dense/dense_1/MatMul/ReadVariableOp?+skip_dense_1/dense_2/BiasAdd/ReadVariableOp?*skip_dense_1/dense_2/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?#@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAdd|
leaky_re_lu/LeakyRelu	LeakyReludense/BiasAdd:output:0*'
_output_shapes
:?????????@2
leaky_re_lu/LeakyRelu?
(skip_dense/dense_1/MatMul/ReadVariableOpReadVariableOp1skip_dense_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02*
(skip_dense/dense_1/MatMul/ReadVariableOp?
skip_dense/dense_1/MatMulMatMul#leaky_re_lu/LeakyRelu:activations:00skip_dense/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
skip_dense/dense_1/MatMul?
)skip_dense/dense_1/BiasAdd/ReadVariableOpReadVariableOp2skip_dense_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)skip_dense/dense_1/BiasAdd/ReadVariableOp?
skip_dense/dense_1/BiasAddBiasAdd#skip_dense/dense_1/MatMul:product:01skip_dense/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
skip_dense/dense_1/BiasAdd?
skip_dense/addAddV2#skip_dense/dense_1/BiasAdd:output:0#leaky_re_lu/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????@2
skip_dense/add|
leaky_re_lu/LeakyRelu_1	LeakyReluskip_dense/add:z:0*'
_output_shapes
:?????????@2
leaky_re_lu/LeakyRelu_1?
*skip_dense_1/dense_2/MatMul/ReadVariableOpReadVariableOp3skip_dense_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02,
*skip_dense_1/dense_2/MatMul/ReadVariableOp?
skip_dense_1/dense_2/MatMulMatMul%leaky_re_lu/LeakyRelu_1:activations:02skip_dense_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
skip_dense_1/dense_2/MatMul?
+skip_dense_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4skip_dense_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+skip_dense_1/dense_2/BiasAdd/ReadVariableOp?
skip_dense_1/dense_2/BiasAddBiasAdd%skip_dense_1/dense_2/MatMul:product:03skip_dense_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
skip_dense_1/dense_2/BiasAdd?
skip_dense_1/addAddV2%skip_dense_1/dense_2/BiasAdd:output:0%leaky_re_lu/LeakyRelu_1:activations:0*
T0*'
_output_shapes
:?????????@2
skip_dense_1/add~
leaky_re_lu/LeakyRelu_2	LeakyReluskip_dense_1/add:z:0*'
_output_shapes
:?????????@2
leaky_re_lu/LeakyRelu_2?
layer_normalization/ShapeShape%leaky_re_lu/LeakyRelu_2:activations:0*
T0*
_output_shapes
:2
layer_normalization/Shape?
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'layer_normalization/strided_slice/stack?
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_1?
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_2?
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!layer_normalization/strided_slicex
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul/x?
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul?
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_1/stack?
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_1?
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_2?
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_1|
layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul_1/x?
layer_normalization/mul_1Mul$layer_normalization/mul_1/x:output:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_1?
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/0?
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/3?
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul:z:0layer_normalization/mul_1:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!layer_normalization/Reshape/shape?
layer_normalization/ReshapeReshape%leaky_re_lu/LeakyRelu_2:activations:0*layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization/Reshape{
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization/Const?
layer_normalization/Fill/dimsPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2
layer_normalization/Fill/dims?
layer_normalization/FillFill&layer_normalization/Fill/dims:output:0"layer_normalization/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization/Fill
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization/Const_1?
layer_normalization/Fill_1/dimsPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization/Fill_1/dims?
layer_normalization/Fill_1Fill(layer_normalization/Fill_1/dims:output:0$layer_normalization/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization/Fill_1}
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_2}
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_3?
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/Fill:output:0#layer_normalization/Fill_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2&
$layer_normalization/FusedBatchNormV3?
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*'
_output_shapes
:?????????@2
layer_normalization/Reshape_1?
'layer_normalization/Cast/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes
:@*
dtype02)
'layer_normalization/Cast/ReadVariableOp?
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:0/layer_normalization/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization/mul_2?
)layer_normalization/Cast_1/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype02+
)layer_normalization/Cast_1/ReadVariableOp?
layer_normalization/addAddV2layer_normalization/mul_2:z:01layer_normalization/Cast_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization/add?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMullayer_normalization/add:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_3/BiasAdd?
leaky_re_lu/LeakyRelu_3	LeakyReludense_3/BiasAdd:output:0*'
_output_shapes
:?????????@2
leaky_re_lu/LeakyRelu_3?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMul%leaky_re_lu/LeakyRelu_3:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddW
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
Equal/y?
EqualEqualinputs_1Equal/y:output:0*
T0*'
_output_shapes
:?????????*
incompatible_shape_error( 2
Equal]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *'?X?2

SelectV2/e?
SelectV2SelectV2	Equal:z:0dense_4/BiasAdd:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:?????????2

SelectV2r
softmax/SoftmaxSoftmaxSelectV2:output:0*
T0*'
_output_shapes
:?????????2
softmax/Softmax?
IdentityIdentitysoftmax/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp(^layer_normalization/Cast/ReadVariableOp*^layer_normalization/Cast_1/ReadVariableOp*^skip_dense/dense_1/BiasAdd/ReadVariableOp)^skip_dense/dense_1/MatMul/ReadVariableOp,^skip_dense_1/dense_2/BiasAdd/ReadVariableOp+^skip_dense_1/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:??????????#:?????????::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2R
'layer_normalization/Cast/ReadVariableOp'layer_normalization/Cast/ReadVariableOp2V
)layer_normalization/Cast_1/ReadVariableOp)layer_normalization/Cast_1/ReadVariableOp2V
)skip_dense/dense_1/BiasAdd/ReadVariableOp)skip_dense/dense_1/BiasAdd/ReadVariableOp2T
(skip_dense/dense_1/MatMul/ReadVariableOp(skip_dense/dense_1/MatMul/ReadVariableOp2Z
+skip_dense_1/dense_2/BiasAdd/ReadVariableOp+skip_dense_1/dense_2/BiasAdd/ReadVariableOp2X
*skip_dense_1/dense_2/MatMul/ReadVariableOp*skip_dense_1/dense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????#
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
1__inference_policy_network_layer_call_fn_34844014
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_policy_network_layer_call_and_return_conditional_losses_348439832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:??????????#:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????#
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
J__inference_skip_dense_1_layer_call_and_return_conditional_losses_34843837
x*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulx%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_2/BiasAddb
addAddV2dense_2/BiasAdd:output:0x*
T0*'
_output_shapes
:?????????@2
add?
IdentityIdentityadd:z:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????@

_user_specified_namex
?	
?
C__inference_dense_layer_call_and_return_conditional_losses_34843769

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?#@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????#::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????#
 
_user_specified_nameinputs
?i
?
__inference_call_1816592
inputs_0
inputs_1(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource5
1skip_dense_dense_1_matmul_readvariableop_resource6
2skip_dense_dense_1_biasadd_readvariableop_resource7
3skip_dense_1_dense_2_matmul_readvariableop_resource8
4skip_dense_1_dense_2_biasadd_readvariableop_resource4
0layer_normalization_cast_readvariableop_resource6
2layer_normalization_cast_1_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?'layer_normalization/Cast/ReadVariableOp?)layer_normalization/Cast_1/ReadVariableOp?)skip_dense/dense_1/BiasAdd/ReadVariableOp?(skip_dense/dense_1/MatMul/ReadVariableOp?+skip_dense_1/dense_2/BiasAdd/ReadVariableOp?*skip_dense_1/dense_2/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?#@*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulinputs_0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2
dense/BiasAddt
leaky_re_lu/LeakyRelu	LeakyReludense/BiasAdd:output:0*
_output_shapes
:	?@2
leaky_re_lu/LeakyRelu?
(skip_dense/dense_1/MatMul/ReadVariableOpReadVariableOp1skip_dense_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02*
(skip_dense/dense_1/MatMul/ReadVariableOp?
skip_dense/dense_1/MatMulMatMul#leaky_re_lu/LeakyRelu:activations:00skip_dense/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2
skip_dense/dense_1/MatMul?
)skip_dense/dense_1/BiasAdd/ReadVariableOpReadVariableOp2skip_dense_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)skip_dense/dense_1/BiasAdd/ReadVariableOp?
skip_dense/dense_1/BiasAddBiasAdd#skip_dense/dense_1/MatMul:product:01skip_dense/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2
skip_dense/dense_1/BiasAdd?
skip_dense/addAddV2#skip_dense/dense_1/BiasAdd:output:0#leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:	?@2
skip_dense/addt
leaky_re_lu/LeakyRelu_1	LeakyReluskip_dense/add:z:0*
_output_shapes
:	?@2
leaky_re_lu/LeakyRelu_1?
*skip_dense_1/dense_2/MatMul/ReadVariableOpReadVariableOp3skip_dense_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02,
*skip_dense_1/dense_2/MatMul/ReadVariableOp?
skip_dense_1/dense_2/MatMulMatMul%leaky_re_lu/LeakyRelu_1:activations:02skip_dense_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2
skip_dense_1/dense_2/MatMul?
+skip_dense_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4skip_dense_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+skip_dense_1/dense_2/BiasAdd/ReadVariableOp?
skip_dense_1/dense_2/BiasAddBiasAdd%skip_dense_1/dense_2/MatMul:product:03skip_dense_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2
skip_dense_1/dense_2/BiasAdd?
skip_dense_1/addAddV2%skip_dense_1/dense_2/BiasAdd:output:0%leaky_re_lu/LeakyRelu_1:activations:0*
T0*
_output_shapes
:	?@2
skip_dense_1/addv
leaky_re_lu/LeakyRelu_2	LeakyReluskip_dense_1/add:z:0*
_output_shapes
:	?@2
leaky_re_lu/LeakyRelu_2?
layer_normalization/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   @   2
layer_normalization/Shape?
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'layer_normalization/strided_slice/stack?
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_1?
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_2?
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!layer_normalization/strided_slicex
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul/x?
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul?
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_1/stack?
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_1?
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_2?
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_1|
layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul_1/x?
layer_normalization/mul_1Mul$layer_normalization/mul_1/x:output:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_1?
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/0?
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/3?
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul:z:0layer_normalization/mul_1:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!layer_normalization/Reshape/shape?
layer_normalization/ReshapeReshape%leaky_re_lu/LeakyRelu_2:activations:0*layer_normalization/Reshape/shape:output:0*
T0*'
_output_shapes
:?@2
layer_normalization/Reshape{
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization/Const?
layer_normalization/Fill/dimsPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2
layer_normalization/Fill/dims?
layer_normalization/FillFill&layer_normalization/Fill/dims:output:0"layer_normalization/Const:output:0*
T0*
_output_shapes	
:?2
layer_normalization/Fill
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization/Const_1?
layer_normalization/Fill_1/dimsPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization/Fill_1/dims?
layer_normalization/Fill_1Fill(layer_normalization/Fill_1/dims:output:0$layer_normalization/Const_1:output:0*
T0*
_output_shapes	
:?2
layer_normalization/Fill_1}
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_2}
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_3?
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/Fill:output:0#layer_normalization/Fill_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*G
_output_shapes5
3:?@:?:?:?:?:*
data_formatNCHW*
epsilon%o?:2&
$layer_normalization/FusedBatchNormV3?
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*
_output_shapes
:	?@2
layer_normalization/Reshape_1?
'layer_normalization/Cast/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes
:@*
dtype02)
'layer_normalization/Cast/ReadVariableOp?
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:0/layer_normalization/Cast/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2
layer_normalization/mul_2?
)layer_normalization/Cast_1/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype02+
)layer_normalization/Cast_1/ReadVariableOp?
layer_normalization/addAddV2layer_normalization/mul_2:z:01layer_normalization/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2
layer_normalization/add?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMullayer_normalization/add:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?@2
dense_3/BiasAddz
leaky_re_lu/LeakyRelu_3	LeakyReludense_3/BiasAdd:output:0*
_output_shapes
:	?@2
leaky_re_lu/LeakyRelu_3?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMul%leaky_re_lu/LeakyRelu_3:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
dense_4/BiasAddW
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
Equal/y}
EqualEqualinputs_1Equal/y:output:0*
T0*
_output_shapes
:	?*
incompatible_shape_error( 2
Equal]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *'?X?2

SelectV2/e?
SelectV2SelectV2	Equal:z:0dense_4/BiasAdd:output:0SelectV2/e:output:0*
T0*
_output_shapes
:	?2

SelectV2j
softmax/SoftmaxSoftmaxSelectV2:output:0*
T0*
_output_shapes
:	?2
softmax/Softmax?
IdentityIdentitysoftmax/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp(^layer_normalization/Cast/ReadVariableOp*^layer_normalization/Cast_1/ReadVariableOp*^skip_dense/dense_1/BiasAdd/ReadVariableOp)^skip_dense/dense_1/MatMul/ReadVariableOp,^skip_dense_1/dense_2/BiasAdd/ReadVariableOp+^skip_dense_1/dense_2/MatMul/ReadVariableOp*
T0*
_output_shapes
:	?2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:
??#:	?::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2R
'layer_normalization/Cast/ReadVariableOp'layer_normalization/Cast/ReadVariableOp2V
)layer_normalization/Cast_1/ReadVariableOp)layer_normalization/Cast_1/ReadVariableOp2V
)skip_dense/dense_1/BiasAdd/ReadVariableOp)skip_dense/dense_1/BiasAdd/ReadVariableOp2T
(skip_dense/dense_1/MatMul/ReadVariableOp(skip_dense/dense_1/MatMul/ReadVariableOp2Z
+skip_dense_1/dense_2/BiasAdd/ReadVariableOp+skip_dense_1/dense_2/BiasAdd/ReadVariableOp2X
*skip_dense_1/dense_2/MatMul/ReadVariableOp*skip_dense_1/dense_2/MatMul/ReadVariableOp:J F
 
_output_shapes
:
??#
"
_user_specified_name
inputs/0:IE

_output_shapes
:	?
"
_user_specified_name
inputs/1
?

/__inference_skip_dense_1_layer_call_fn_34844214
x
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_skip_dense_1_layer_call_and_return_conditional_losses_348438372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????@

_user_specified_namex
?
?
J__inference_skip_dense_1_layer_call_and_return_conditional_losses_34844205
x*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulx%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_2/BiasAddb
addAddV2dense_2/BiasAdd:output:0x*
T0*'
_output_shapes
:?????????@2
add?
IdentityIdentityadd:z:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????@

_user_specified_namex
?
e
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_34843790

inputs
identityT
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????@2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?$
?
!__inference__traced_save_34844274
file_prefix8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop8
4savev2_skip_dense_dense_1_kernel_read_readvariableop6
2savev2_skip_dense_dense_1_bias_read_readvariableop:
6savev2_skip_dense_1_dense_2_kernel_read_readvariableop8
4savev2_skip_dense_1_dense_2_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B.normalization/gamma/.ATTRIBUTES/VARIABLE_VALUEB-normalization/beta/.ATTRIBUTES/VARIABLE_VALUEB+lastlayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB)lastlayer/bias/.ATTRIBUTES/VARIABLE_VALUEB+out_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB)out_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop4savev2_skip_dense_dense_1_kernel_read_readvariableop2savev2_skip_dense_dense_1_bias_read_readvariableop6savev2_skip_dense_1_dense_2_kernel_read_readvariableop4savev2_skip_dense_1_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*t
_input_shapesc
a: :@:@:@@:@:@::	?#@:@:@@:@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	?#@: 

_output_shapes
:@:$	 

_output_shapes

:@@: 


_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:

_output_shapes
: 
?5
?
L__inference_policy_network_layer_call_and_return_conditional_losses_34843983
input_1
input_2
dense_34843780
dense_34843782
skip_dense_34843820
skip_dense_34843822
skip_dense_1_34843848
skip_dense_1_34843850 
layer_normalization_34843907 
layer_normalization_34843909
dense_3_34843933
dense_3_34843935
dense_4_34843960
dense_4_34843962
identity??dense/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?+layer_normalization/StatefulPartitionedCall?"skip_dense/StatefulPartitionedCall?$skip_dense_1/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_34843780dense_34843782*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_348437692
dense/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_348437902
leaky_re_lu/PartitionedCall?
"skip_dense/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0skip_dense_34843820skip_dense_34843822*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_skip_dense_layer_call_and_return_conditional_losses_348438092$
"skip_dense/StatefulPartitionedCall?
leaky_re_lu/PartitionedCall_1PartitionedCall+skip_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_348437902
leaky_re_lu/PartitionedCall_1?
$skip_dense_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu/PartitionedCall_1:output:0skip_dense_1_34843848skip_dense_1_34843850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_skip_dense_1_layer_call_and_return_conditional_losses_348438372&
$skip_dense_1/StatefulPartitionedCall?
leaky_re_lu/PartitionedCall_2PartitionedCall-skip_dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_348437902
leaky_re_lu/PartitionedCall_2?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu/PartitionedCall_2:output:0layer_normalization_34843907layer_normalization_34843909*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_348438962-
+layer_normalization/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0dense_3_34843933dense_3_34843935*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_348439222!
dense_3/StatefulPartitionedCall?
leaky_re_lu/PartitionedCall_3PartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_348437902
leaky_re_lu/PartitionedCall_3?
dense_4/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu/PartitionedCall_3:output:0dense_4_34843960dense_4_34843962*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_348439492!
dense_4/StatefulPartitionedCallW
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
Equal/y?
EqualEqualinput_2Equal/y:output:0*
T0*'
_output_shapes
:?????????*
incompatible_shape_error( 2
Equal]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *'?X?2

SelectV2/e?
SelectV2SelectV2	Equal:z:0(dense_4/StatefulPartitionedCall:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:?????????2

SelectV2?
softmax/PartitionedCallPartitionedCallSelectV2:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_348439742
softmax/PartitionedCall?
IdentityIdentity softmax/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall#^skip_dense/StatefulPartitionedCall%^skip_dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:??????????#:?????????::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2H
"skip_dense/StatefulPartitionedCall"skip_dense/StatefulPartitionedCall2L
$skip_dense_1/StatefulPartitionedCall$skip_dense_1/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????#
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
J
.__inference_leaky_re_lu_layer_call_fn_34844056

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_348437902
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
#__inference__wrapped_model_34843754
input_1
input_2
policy_network_34843728
policy_network_34843730
policy_network_34843732
policy_network_34843734
policy_network_34843736
policy_network_34843738
policy_network_34843740
policy_network_34843742
policy_network_34843744
policy_network_34843746
policy_network_34843748
policy_network_34843750
identity??&policy_network/StatefulPartitionedCall?
&policy_network/StatefulPartitionedCallStatefulPartitionedCallinput_1input_2policy_network_34843728policy_network_34843730policy_network_34843732policy_network_34843734policy_network_34843736policy_network_34843738policy_network_34843740policy_network_34843742policy_network_34843744policy_network_34843746policy_network_34843748policy_network_34843750*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *!
fR
__inference_call_18161892(
&policy_network/StatefulPartitionedCall?
IdentityIdentity/policy_network/StatefulPartitionedCall:output:0'^policy_network/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:??????????#:?????????::::::::::::2P
&policy_network/StatefulPartitionedCall&policy_network/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????#
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?	
?
E__inference_dense_3_layer_call_and_return_conditional_losses_34843922

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
&__inference_signature_wrapper_34844046
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_348437542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:??????????#:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????#
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?5
?
$__inference__traced_restore_34844320
file_prefix.
*assignvariableop_layer_normalization_gamma/
+assignvariableop_1_layer_normalization_beta%
!assignvariableop_2_dense_3_kernel#
assignvariableop_3_dense_3_bias%
!assignvariableop_4_dense_4_kernel#
assignvariableop_5_dense_4_bias#
assignvariableop_6_dense_kernel!
assignvariableop_7_dense_bias0
,assignvariableop_8_skip_dense_dense_1_kernel.
*assignvariableop_9_skip_dense_dense_1_bias3
/assignvariableop_10_skip_dense_1_dense_2_kernel1
-assignvariableop_11_skip_dense_1_dense_2_bias
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B.normalization/gamma/.ATTRIBUTES/VARIABLE_VALUEB-normalization/beta/.ATTRIBUTES/VARIABLE_VALUEB+lastlayer/kernel/.ATTRIBUTES/VARIABLE_VALUEB)lastlayer/bias/.ATTRIBUTES/VARIABLE_VALUEB+out_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB)out_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_layer_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp,assignvariableop_8_skip_dense_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_skip_dense_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_skip_dense_1_dense_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp-assignvariableop_11_skip_dense_1_dense_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12?
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
E__inference_dense_4_layer_call_and_return_conditional_losses_34844146

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
F
*__inference_softmax_layer_call_fn_34844066

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_348439742
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

*__inference_dense_3_layer_call_fn_34844136

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_348439222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
}
-__inference_skip_dense_layer_call_fn_34844194
x
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_skip_dense_layer_call_and_return_conditional_losses_348438092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????@

_user_specified_namex
?
e
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_34844051

inputs
identityT
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????@2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
H__inference_skip_dense_layer_call_and_return_conditional_losses_34844185
x*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulx%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/BiasAddb
addAddV2dense_1/BiasAdd:output:0x*
T0*'
_output_shapes
:?????????@2
add?
IdentityIdentityadd:z:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????@

_user_specified_namex
?"
?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_34843896

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
ReshapeS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstU
	Fill/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
	Fill/dimsf
FillFillFill/dims:output:0Const:output:0*
T0*#
_output_shapes
:?????????2
FillW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1Y
Fill_1/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
Fill_1/dimsn
Fill_1FillFill_1/dims:output:0Const_1:output:0*
T0*#
_output_shapes
:?????????2
Fill_1U
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_2U
Const_3Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_3?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0Fill:output:0Fill_1:output:0Const_2:output:0Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????@2
	Reshape_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast/ReadVariableOpx
mul_2MulReshape_1:output:0Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
mul_2?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast_1/ReadVariableOpo
addAddV2	mul_2:z:0Cast_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
add?
IdentityIdentityadd:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?k
?
__inference_call_1816676
inputs_0
inputs_1(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource5
1skip_dense_dense_1_matmul_readvariableop_resource6
2skip_dense_dense_1_biasadd_readvariableop_resource7
3skip_dense_1_dense_2_matmul_readvariableop_resource8
4skip_dense_1_dense_2_biasadd_readvariableop_resource4
0layer_normalization_cast_readvariableop_resource6
2layer_normalization_cast_1_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?'layer_normalization/Cast/ReadVariableOp?)layer_normalization/Cast_1/ReadVariableOp?)skip_dense/dense_1/BiasAdd/ReadVariableOp?(skip_dense/dense_1/MatMul/ReadVariableOp?+skip_dense_1/dense_2/BiasAdd/ReadVariableOp?*skip_dense_1/dense_2/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?#@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAdd|
leaky_re_lu/LeakyRelu	LeakyReludense/BiasAdd:output:0*'
_output_shapes
:?????????@2
leaky_re_lu/LeakyRelu?
(skip_dense/dense_1/MatMul/ReadVariableOpReadVariableOp1skip_dense_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02*
(skip_dense/dense_1/MatMul/ReadVariableOp?
skip_dense/dense_1/MatMulMatMul#leaky_re_lu/LeakyRelu:activations:00skip_dense/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
skip_dense/dense_1/MatMul?
)skip_dense/dense_1/BiasAdd/ReadVariableOpReadVariableOp2skip_dense_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)skip_dense/dense_1/BiasAdd/ReadVariableOp?
skip_dense/dense_1/BiasAddBiasAdd#skip_dense/dense_1/MatMul:product:01skip_dense/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
skip_dense/dense_1/BiasAdd?
skip_dense/addAddV2#skip_dense/dense_1/BiasAdd:output:0#leaky_re_lu/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????@2
skip_dense/add|
leaky_re_lu/LeakyRelu_1	LeakyReluskip_dense/add:z:0*'
_output_shapes
:?????????@2
leaky_re_lu/LeakyRelu_1?
*skip_dense_1/dense_2/MatMul/ReadVariableOpReadVariableOp3skip_dense_1_dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02,
*skip_dense_1/dense_2/MatMul/ReadVariableOp?
skip_dense_1/dense_2/MatMulMatMul%leaky_re_lu/LeakyRelu_1:activations:02skip_dense_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
skip_dense_1/dense_2/MatMul?
+skip_dense_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4skip_dense_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+skip_dense_1/dense_2/BiasAdd/ReadVariableOp?
skip_dense_1/dense_2/BiasAddBiasAdd%skip_dense_1/dense_2/MatMul:product:03skip_dense_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
skip_dense_1/dense_2/BiasAdd?
skip_dense_1/addAddV2%skip_dense_1/dense_2/BiasAdd:output:0%leaky_re_lu/LeakyRelu_1:activations:0*
T0*'
_output_shapes
:?????????@2
skip_dense_1/add~
leaky_re_lu/LeakyRelu_2	LeakyReluskip_dense_1/add:z:0*'
_output_shapes
:?????????@2
leaky_re_lu/LeakyRelu_2?
layer_normalization/ShapeShape%leaky_re_lu/LeakyRelu_2:activations:0*
T0*
_output_shapes
:2
layer_normalization/Shape?
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'layer_normalization/strided_slice/stack?
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_1?
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_2?
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!layer_normalization/strided_slicex
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul/x?
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul?
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_1/stack?
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_1?
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_2?
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_1|
layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul_1/x?
layer_normalization/mul_1Mul$layer_normalization/mul_1/x:output:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_1?
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/0?
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/3?
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul:z:0layer_normalization/mul_1:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!layer_normalization/Reshape/shape?
layer_normalization/ReshapeReshape%leaky_re_lu/LeakyRelu_2:activations:0*layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2
layer_normalization/Reshape{
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_normalization/Const?
layer_normalization/Fill/dimsPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2
layer_normalization/Fill/dims?
layer_normalization/FillFill&layer_normalization/Fill/dims:output:0"layer_normalization/Const:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization/Fill
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization/Const_1?
layer_normalization/Fill_1/dimsPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization/Fill_1/dims?
layer_normalization/Fill_1Fill(layer_normalization/Fill_1/dims:output:0$layer_normalization/Const_1:output:0*
T0*#
_output_shapes
:?????????2
layer_normalization/Fill_1}
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_2}
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_3?
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/Fill:output:0#layer_normalization/Fill_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2&
$layer_normalization/FusedBatchNormV3?
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*'
_output_shapes
:?????????@2
layer_normalization/Reshape_1?
'layer_normalization/Cast/ReadVariableOpReadVariableOp0layer_normalization_cast_readvariableop_resource*
_output_shapes
:@*
dtype02)
'layer_normalization/Cast/ReadVariableOp?
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:0/layer_normalization/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization/mul_2?
)layer_normalization/Cast_1/ReadVariableOpReadVariableOp2layer_normalization_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype02+
)layer_normalization/Cast_1/ReadVariableOp?
layer_normalization/addAddV2layer_normalization/mul_2:z:01layer_normalization/Cast_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
layer_normalization/add?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMullayer_normalization/add:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_3/BiasAdd?
leaky_re_lu/LeakyRelu_3	LeakyReludense_3/BiasAdd:output:0*'
_output_shapes
:?????????@2
leaky_re_lu/LeakyRelu_3?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMul%leaky_re_lu/LeakyRelu_3:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddW
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
Equal/y?
EqualEqualinputs_1Equal/y:output:0*
T0*'
_output_shapes
:?????????*
incompatible_shape_error( 2
Equal]

SelectV2/eConst*
_output_shapes
: *
dtype0*
valueB
 *'?X?2

SelectV2/e?
SelectV2SelectV2	Equal:z:0dense_4/BiasAdd:output:0SelectV2/e:output:0*
T0*'
_output_shapes
:?????????2

SelectV2r
softmax/SoftmaxSoftmaxSelectV2:output:0*
T0*'
_output_shapes
:?????????2
softmax/Softmax?
IdentityIdentitysoftmax/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp(^layer_normalization/Cast/ReadVariableOp*^layer_normalization/Cast_1/ReadVariableOp*^skip_dense/dense_1/BiasAdd/ReadVariableOp)^skip_dense/dense_1/MatMul/ReadVariableOp,^skip_dense_1/dense_2/BiasAdd/ReadVariableOp+^skip_dense_1/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*j
_input_shapesY
W:??????????#:?????????::::::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2R
'layer_normalization/Cast/ReadVariableOp'layer_normalization/Cast/ReadVariableOp2V
)layer_normalization/Cast_1/ReadVariableOp)layer_normalization/Cast_1/ReadVariableOp2V
)skip_dense/dense_1/BiasAdd/ReadVariableOp)skip_dense/dense_1/BiasAdd/ReadVariableOp2T
(skip_dense/dense_1/MatMul/ReadVariableOp(skip_dense/dense_1/MatMul/ReadVariableOp2Z
+skip_dense_1/dense_2/BiasAdd/ReadVariableOp+skip_dense_1/dense_2/BiasAdd/ReadVariableOp2X
*skip_dense_1/dense_2/MatMul/ReadVariableOp*skip_dense_1/dense_2/MatMul/ReadVariableOp:R N
(
_output_shapes
:??????????#
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

*__inference_dense_4_layer_call_fn_34844155

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_348439492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?"
?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_34844108

inputs 
cast_readvariableop_resource"
cast_1_readvariableop_resource
identity??Cast/ReadVariableOp?Cast_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceP
mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
mul/xZ
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: 2
mulx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1T
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/xb
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: 2
mul_1d
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/0d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape?
ReshapeReshapeinputsReshape/shape:output:0*
T0*8
_output_shapes&
$:"??????????????????2	
ReshapeS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstU
	Fill/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
	Fill/dimsf
FillFillFill/dims:output:0Const:output:0*
T0*#
_output_shapes
:?????????2
FillW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1Y
Fill_1/dimsPackmul:z:0*
N*
T0*
_output_shapes
:2
Fill_1/dimsn
Fill_1FillFill_1/dims:output:0Const_1:output:0*
T0*#
_output_shapes
:?????????2
Fill_1U
Const_2Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_2U
Const_3Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_3?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0Fill:output:0Fill_1:output:0Const_2:output:0Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"??????????????????:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:2
FusedBatchNormV3y
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????@2
	Reshape_1?
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast/ReadVariableOpx
mul_2MulReshape_1:output:0Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
mul_2?
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
:@*
dtype02
Cast_1/ReadVariableOpo
addAddV2	mul_2:z:0Cast_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
add?
IdentityIdentityadd:z:0^Cast/ReadVariableOp^Cast_1/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
H__inference_skip_dense_layer_call_and_return_conditional_losses_34843809
x*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulx%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/BiasAddb
addAddV2dense_1/BiasAdd:output:0x*
T0*'
_output_shapes
:?????????@2
add?
IdentityIdentityadd:z:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????@

_user_specified_namex
?
}
(__inference_dense_layer_call_fn_34844174

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_348437692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????#::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????#
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????#
;
input_20
serving_default_input_2:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?

activation
softmax

hidden
normalization
	lastlayer
	out_layer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
}_default_save_signature
*~&call_and_return_all_conditional_losses
__call__
	?call"?
_tf_keras_model?{"class_name": "PolicyNetwork", "name": "policy_network", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "PolicyNetwork"}}
?
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
?
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Softmax", "name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": -1}}
5
0
1
2"
trackable_list_wrapper
?
axis
	gamma
beta
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LayerNormalization", "name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [1], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [2048, 64]}}
?

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [2048, 64]}}
?

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [2048, 64]}}
v
*0
+1
,2
-3
.4
/5
6
7
8
9
$10
%11"
trackable_list_wrapper
v
*0
+1
,2
-3
.4
/5
6
7
8
9
$10
%11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0layer_regularization_losses
	variables
trainable_variables
1layer_metrics
2metrics
	regularization_losses

3layers
4non_trainable_variables
__call__
}_default_save_signature
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
5layer_regularization_losses
	variables
trainable_variables
6layer_metrics
7metrics
regularization_losses

8layers
9non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
:layer_regularization_losses
	variables
trainable_variables
;layer_metrics
<metrics
regularization_losses

=layers
>non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

*kernel
+bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4498}}}, "build_input_shape": {"class_name": "TensorShape", "items": [2048, 4498]}}
?

Chidden
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "SkipDense", "name": "skip_dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?

Hhidden
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "SkipDense", "name": "skip_dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
 "
trackable_list_wrapper
':%@2layer_normalization/gamma
&:$@2layer_normalization/beta
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Mlayer_regularization_losses
	variables
trainable_variables
Nlayer_metrics
Ometrics
regularization_losses

Players
Qnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@@2dense_3/kernel
:@2dense_3/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rlayer_regularization_losses
 	variables
!trainable_variables
Slayer_metrics
Tmetrics
"regularization_losses

Ulayers
Vnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_4/kernel
:2dense_4/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wlayer_regularization_losses
&	variables
'trainable_variables
Xlayer_metrics
Ymetrics
(regularization_losses

Zlayers
[non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?#@2dense/kernel
:@2
dense/bias
+:)@@2skip_dense/dense_1/kernel
%:#@2skip_dense/dense_1/bias
-:+@@2skip_dense_1/dense_2/kernel
':%@2skip_dense_1/dense_2/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\layer_regularization_losses
?	variables
@trainable_variables
]layer_metrics
^metrics
Aregularization_losses

_layers
`non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

,kernel
-bias
a	variables
btrainable_variables
cregularization_losses
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [2048, 64]}}
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
elayer_regularization_losses
D	variables
Etrainable_variables
flayer_metrics
gmetrics
Fregularization_losses

hlayers
inon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

.kernel
/bias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [2048, 64]}}
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
nlayer_regularization_losses
I	variables
Jtrainable_variables
olayer_metrics
pmetrics
Kregularization_losses

qlayers
rnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
slayer_regularization_losses
a	variables
btrainable_variables
tlayer_metrics
umetrics
cregularization_losses

vlayers
wnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
xlayer_regularization_losses
j	variables
ktrainable_variables
ylayer_metrics
zmetrics
lregularization_losses

{layers
|non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
H0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
#__inference__wrapped_model_34843754?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *O?L
J?G
"?
input_1??????????#
!?
input_2?????????
?2?
L__inference_policy_network_layer_call_and_return_conditional_losses_34843983?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *O?L
J?G
"?
input_1??????????#
!?
input_2?????????
?2?
1__inference_policy_network_layer_call_fn_34844014?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *O?L
J?G
"?
input_1??????????#
!?
input_2?????????
?2?
__inference_call_1816592
__inference_call_1816676?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_34844051?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_leaky_re_lu_layer_call_fn_34844056?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_softmax_layer_call_and_return_conditional_losses_34844061?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_softmax_layer_call_fn_34844066?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_34844108?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_layer_normalization_layer_call_fn_34844117?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_3_layer_call_and_return_conditional_losses_34844127?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_3_layer_call_fn_34844136?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_4_layer_call_and_return_conditional_losses_34844146?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_4_layer_call_fn_34844155?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_34844046input_1input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_layer_call_and_return_conditional_losses_34844165?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_layer_call_fn_34844174?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_skip_dense_layer_call_and_return_conditional_losses_34844185?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_skip_dense_layer_call_fn_34844194?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_skip_dense_1_layer_call_and_return_conditional_losses_34844205?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_skip_dense_1_layer_call_fn_34844214?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_34843754?*+,-./$%Y?V
O?L
J?G
"?
input_1??????????#
!?
input_2?????????
? "3?0
.
output_1"?
output_1??????????
__inference_call_1816592m*+,-./$%K?H
A?>
<?9
?
inputs/0
??#
?
inputs/1	?
? "?	??
__inference_call_1816676?*+,-./$%[?X
Q?N
L?I
#? 
inputs/0??????????#
"?
inputs/1?????????
? "???????????
E__inference_dense_3_layer_call_and_return_conditional_losses_34844127\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? }
*__inference_dense_3_layer_call_fn_34844136O/?,
%?"
 ?
inputs?????????@
? "??????????@?
E__inference_dense_4_layer_call_and_return_conditional_losses_34844146\$%/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? }
*__inference_dense_4_layer_call_fn_34844155O$%/?,
%?"
 ?
inputs?????????@
? "???????????
C__inference_dense_layer_call_and_return_conditional_losses_34844165]*+0?-
&?#
!?
inputs??????????#
? "%?"
?
0?????????@
? |
(__inference_dense_layer_call_fn_34844174P*+0?-
&?#
!?
inputs??????????#
? "??????????@?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_34844108\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ?
6__inference_layer_normalization_layer_call_fn_34844117O/?,
%?"
 ?
inputs?????????@
? "??????????@?
I__inference_leaky_re_lu_layer_call_and_return_conditional_losses_34844051X/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? }
.__inference_leaky_re_lu_layer_call_fn_34844056K/?,
%?"
 ?
inputs?????????@
? "??????????@?
L__inference_policy_network_layer_call_and_return_conditional_losses_34843983?*+,-./$%Y?V
O?L
J?G
"?
input_1??????????#
!?
input_2?????????
? "%?"
?
0?????????
? ?
1__inference_policy_network_layer_call_fn_34844014?*+,-./$%Y?V
O?L
J?G
"?
input_1??????????#
!?
input_2?????????
? "???????????
&__inference_signature_wrapper_34844046?*+,-./$%j?g
? 
`?]
-
input_1"?
input_1??????????#
,
input_2!?
input_2?????????"3?0
.
output_1"?
output_1??????????
J__inference_skip_dense_1_layer_call_and_return_conditional_losses_34844205W./*?'
 ?
?
x?????????@
? "%?"
?
0?????????@
? }
/__inference_skip_dense_1_layer_call_fn_34844214J./*?'
 ?
?
x?????????@
? "??????????@?
H__inference_skip_dense_layer_call_and_return_conditional_losses_34844185W,-*?'
 ?
?
x?????????@
? "%?"
?
0?????????@
? {
-__inference_skip_dense_layer_call_fn_34844194J,-*?'
 ?
?
x?????????@
? "??????????@?
E__inference_softmax_layer_call_and_return_conditional_losses_34844061\3?0
)?&
 ?
inputs?????????

 
? "%?"
?
0?????????
? }
*__inference_softmax_layer_call_fn_34844066O3?0
)?&
 ?
inputs?????????

 
? "??????????