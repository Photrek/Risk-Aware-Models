import React from "react";
import Grid from "@material-ui/core/Grid";
import Button from "@material-ui/core/Button";

import OutlinedDropDown from "../../common/OutlinedDropdown";
import OutlinedTextArea from "../../common/OutlinedTextArea";

import { ServiceDefinition } from "./data_generator_pb_service";

const initialUserInput = {
  dataType: [
    { 
      label: "MNIST",
      value: "MNIST",
    },
    { 
      label: "CIFAR10",
      value: "CIFAR",
    },
  ],
  class: {
    MNIST: [
      { 
        label: "Zero",
        value: "0",
      },
      { 
        label: "One",
        value: "1",
      },
      { 
        label: "Two",
        value: "2",
      },
      { 
        label: "Three",
        value: "3",
      },
      { 
        label: "Four",
        value: "4",
      },
      { 
        label: "Five",
        value: "5",
      },
      { 
        label: "Six",
        value: "6",
      },
      { 
        label: "Seven",
        value: "7",
      },
      { 
        label: "Eight",
        value: "8",
      },
      { 
        label: "Nine",
        value: "9",
      }
    ],
    CIFAR: [
      { 
        label: "Airplane",
        value: "0",
      },
      { 
        label: "Automobile",
        value: "1",
      },
      { 
        label: "Bird",
        value: "2",
      },
      { 
        label: "Cat",
        value: "3",
      },
      { 
        label: "Deer",
        value: "4",
      },
      { 
        label: "Dog",
        value: "5",
      },
      { 
        label: "Frog",
        value: "6",
      },
      { 
        label: "Horse",
        value: "7",
      },
      { 
        label: "Ship",
        value: "8",
      },
      { 
        label: "Truck",
        value: "9",
      }
    ]
  },
  d: "MNIST",
  k: "0"
};

export default class CVAE extends React.Component {
  constructor(props) {
    super(props);
    this.submitAction = this.submitAction.bind(this);
    this.handleFormUpdate = this.handleFormUpdate.bind(this);
    this.state = { ...initialUserInput };
  }

  handleFormUpdate(event) {
    this.setState({
      [event.target.name]: event.target.value,
    });
  }

  onKeyPressvalidator(event) {
    const keyCode = event.keyCode || event.which;
    if (!(keyCode === 8 || keyCode === 46) && (keyCode < 48 || keyCode > 57)) {
      event.preventDefault();
    } else {
      let dots = event.target.value.split(".");
      if (dots.length > 1 && keyCode === 46) event.preventDefault();
    }
  }

  submitAction() {
    const methodDescriptor = ServiceDefinition["GenerateImage"];
    const request = new methodDescriptor.requestType();

    console.log(this.state.d)
    //Following confirmed defined in adr_pb.js
    request.setK(this.state.k);
    request.setD(this.state.d);

    const props = {
      request,
      onEnd: ({ message }) => {
        //console.log("DEBUGGER ==========");debugger;
        //this.setState({ ...initialUserInput, response: { value: message.getS() } });
        //this.setState({ ...initialUserInput, response: { a: message.getA(), d: message.getD(), r: message.getR() } });
        //this.setState({ ...initialUserInput, response: "a: "+toString(message.getA()) + ", d:" + toString(message.getD()) + ", r:"+toString(message.getR()) });
        // this.setState({ ...initialUserInput, response: "a: "+message.getA().toString() + ", d:" + message.getD().toString() + ", r:"+message.getR().toString() });
        let imgb64 = message.getS();
        imgb64.length % 4 > 0 && (imgb64 += "=".repeat(4 - (imgb64.length % 4)));
        console.log("<<<<<<<<<<<<<<<<< submitAction: img = "+imgb64.slice(0,10)+" (...) "+imgb64.slice(-10));
        this.setState({ ...initialUserInput, response: { 
          img: imgb64
      } });
      },
    };
    this.props.serviceClient.unary(methodDescriptor, props);
  }

  handleFocus = event => event.target.select();

  renderForm() {
    return (
      <React.Fragment>
        <Grid container direction="column" alignItems="center" justify="center">
          <Grid item xs={6} container spacing={1}>
            <Grid item xs>
              <OutlinedDropDown
                id="datatype"
                name="d"
                label="Data Type"
                fullWidth={true}
                list={this.state.dataType}
                value={this.state.d}
                onChange={this.handleFormUpdate}
                onKeyPress={e => this.onKeyPressvalidator(e)}
                onFocus={this.handleFocus}
              />
              <OutlinedDropDown
                id="input_k"
                name="k"
                label="Class"
                fullWidth={true}
                list={this.state.class[this.state.d]}
                value={this.state.k}
                onChange={this.handleFormUpdate}
                onKeyPress={e => this.onKeyPressvalidator(e)}
                onFocus={this.handleFocus}
              />
              {/* <OutlinedTextArea
                id="input_k"
                name="k"
                label="Class"
                type="text"
                fullWidth={false}
                value={this.state.k}
                rows={1}
                onChange={this.handleFormUpdate}
                onKeyPress={e => this.onKeyPressvalidator(e)}
                onFocus={this.handleFocus}
              /> */}
            </Grid>
          </Grid>

          <Grid item xs={12} style={{ textAlign: "center" }}>
            <Button variant="contained" color="primary" onClick={this.submitAction}>
              Invoke
            </Button>
          </Grid>
        </Grid>
      </React.Fragment>
    );
  }


  parseResponse() {
    const { response } = this.state;
    if (typeof response !== "undefined") {
      if (typeof response === "string") {
        //return response;
        return (
          <div>
            <h4>Risk Aware Assessment Results</h4>
            <div style={{padding: "10px 10px 0 10px", fontSize: "14px", color:"#9b9b9b"}}>
              Results: <span style={{color: "#222222"}}>{response}</span>
            </div>
          </div>
        );
      } else if (typeof response === "object") {
        //let imgsrc='"data:img/png;base64,'+response.img+'"';
        let imgsrc="data:img/png;base64,"+response.img;
        const atsymbol = '&#64;';
        console.log("<<<<< imgsrc: "+imgsrc.slice(0,25)+" (...) "+imgsrc.slice(-25));
        return (
          <div>
            <h4>Data Generator Results</h4>
            <div style={{padding: "0 10px 0 10px", fontSize: "14px", color:"#777777"}}>
              <center><img src={imgsrc} width="400" text-align='center' display='block'/></center>
              <br/>
Thank you for using Risk Aware Assessment by <a href="https://photrek.io">Photrek</a>. Should you have experienced difficulty using or understanding results of this service, please contact us at <a href="mailto:kenric.nelson@photrek.io">kenric.nelson@photrek.io</a>.
            </div>
          </div>
        );
      }
      //console.log("<<<<<<<<<<<<<<<<<<<<< Response type: "+typeof response);
      //return response.value;
    }
  }

  renderComplete() {
    const response = this.parseResponse();
    return (
      <Grid item xs={12} container justify="center">
        <p style={{ fontSize: "20px" }}>Response from service: {response} </p>
      </Grid>
    );
  }

  render() {
    if (this.props.isComplete) return <div>{this.renderComplete()}</div>;
    else {
      return <div>{this.renderForm()}</div>;
    }
  }
}