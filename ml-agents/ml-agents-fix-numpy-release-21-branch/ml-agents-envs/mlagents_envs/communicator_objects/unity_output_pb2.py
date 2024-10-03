# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mlagents_envs/communicator_objects/unity_output.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mlagents_envs.communicator_objects import unity_rl_output_pb2 as mlagents__envs_dot_communicator__objects_dot_unity__rl__output__pb2
from mlagents_envs.communicator_objects import unity_rl_initialization_output_pb2 as mlagents__envs_dot_communicator__objects_dot_unity__rl__initialization__output__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='mlagents_envs/communicator_objects/unity_output.proto',
  package='communicator_objects',
  syntax='proto3',
  serialized_pb=_b('\n5mlagents_envs/communicator_objects/unity_output.proto\x12\x14\x63ommunicator_objects\x1a\x38mlagents_envs/communicator_objects/unity_rl_output.proto\x1aGmlagents_envs/communicator_objects/unity_rl_initialization_output.proto\"\xa9\x01\n\x10UnityOutputProto\x12;\n\trl_output\x18\x01 \x01(\x0b\x32(.communicator_objects.UnityRLOutputProto\x12X\n\x18rl_initialization_output\x18\x02 \x01(\x0b\x32\x36.communicator_objects.UnityRLInitializationOutputProtoB%\xaa\x02\"Unity.MLAgents.CommunicatorObjectsb\x06proto3')
  ,
  dependencies=[mlagents__envs_dot_communicator__objects_dot_unity__rl__output__pb2.DESCRIPTOR,mlagents__envs_dot_communicator__objects_dot_unity__rl__initialization__output__pb2.DESCRIPTOR,])




_UNITYOUTPUTPROTO = _descriptor.Descriptor(
  name='UnityOutputProto',
  full_name='communicator_objects.UnityOutputProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='rl_output', full_name='communicator_objects.UnityOutputProto.rl_output', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rl_initialization_output', full_name='communicator_objects.UnityOutputProto.rl_initialization_output', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=211,
  serialized_end=380,
)

_UNITYOUTPUTPROTO.fields_by_name['rl_output'].message_type = mlagents__envs_dot_communicator__objects_dot_unity__rl__output__pb2._UNITYRLOUTPUTPROTO
_UNITYOUTPUTPROTO.fields_by_name['rl_initialization_output'].message_type = mlagents__envs_dot_communicator__objects_dot_unity__rl__initialization__output__pb2._UNITYRLINITIALIZATIONOUTPUTPROTO
DESCRIPTOR.message_types_by_name['UnityOutputProto'] = _UNITYOUTPUTPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

UnityOutputProto = _reflection.GeneratedProtocolMessageType('UnityOutputProto', (_message.Message,), dict(
  DESCRIPTOR = _UNITYOUTPUTPROTO,
  __module__ = 'mlagents_envs.communicator_objects.unity_output_pb2'
  # @@protoc_insertion_point(class_scope:communicator_objects.UnityOutputProto)
  ))
_sym_db.RegisterMessage(UnityOutputProto)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\252\002\"Unity.MLAgents.CommunicatorObjects'))
# @@protoc_insertion_point(module_scope)
