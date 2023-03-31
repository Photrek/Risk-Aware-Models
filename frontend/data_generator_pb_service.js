// package: data_generator
// file: data_generator.proto

var data_generator_pb = require("./data_generator_pb");
var grpc = require("@improbable-eng/grpc-web").grpc;

var ServiceDefinition = (function () {
  function ServiceDefinition() {}
  ServiceDefinition.serviceName = "data_generator.ServiceDefinition";
  return ServiceDefinition;
}());

ServiceDefinition.GenerateImage = {
  methodName: "GenerateImage",
  service: ServiceDefinition,
  requestStream: false,
  responseStream: false,
  requestType: data_generator_pb.Input,
  responseType: data_generator_pb.StringResponse
};

exports.ServiceDefinition = ServiceDefinition;

function ServiceDefinitionClient(serviceHost, options) {
  this.serviceHost = serviceHost;
  this.options = options || {};
}

ServiceDefinitionClient.prototype.generateImage = function generateImage(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(ServiceDefinition.GenerateImage, {
    request: requestMessage,
    host: this.serviceHost,
    metadata: metadata,
    transport: this.options.transport,
    debug: this.options.debug,
    onEnd: function (response) {
      if (callback) {
        if (response.status !== grpc.Code.OK) {
          var err = new Error(response.statusMessage);
          err.code = response.status;
          err.metadata = response.trailers;
          callback(err, null);
        } else {
          callback(null, response.message);
        }
      }
    }
  });
  return {
    cancel: function () {
      callback = null;
      client.close();
    }
  };
};

exports.ServiceDefinitionClient = ServiceDefinitionClient;

