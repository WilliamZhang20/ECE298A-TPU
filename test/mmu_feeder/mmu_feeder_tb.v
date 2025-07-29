`default_nettype none
`timescale 1ns / 1ps

/* This testbench just instantiates the module and makes some convenient wires
   that can be driven / tested by the cocotb test.py.
*/
module mmu_feeder_tb ();

// Dump the signals to a VCD file. You can view it with gtkwave or surfer.
	initial begin
`ifdef VCD_PATH
		$dumpfile(`VCD_PATH);
`else
		$dumpfile("mmu_feeder_tb.vcd");
`endif
		$dumpvars(0, mmu_feeder_tb);
		#1;
	end

  wire clk;
  wire rst;
  wire en;
  wire [2:0] mmu_cycle,

  /* Memory module interface */
  wire [7:0] weight0, weight1, weight2, weight3;

  wire [7:0] input0, input1, input2, input3;

  /*  mmu -> feeder  */
  wire signed [15:0] c00, c01, c10, c11;

  /*  feeder -> mmu */
  wire clear;
  wire [7:0] a_data0;
  wire [7:0] a_data1;
  wire [7:0] b_data0;
  wire [7:0] b_data1;

  /*  feeder -> rpi */
  wire done;
  reg [7:0] host_outdata;

  // Instantiate the MMU feeder module
  mmu_feeder dut (
    .clk(clk),
    .rst(rst),
    .en(en),
    .mmu_cycle(mmu_cycle),

    .weight0(weight0),
    .weight1(weight1),
    .weight2(weight2),
    .weight3(weight3),

    .input0(input0),
    .input1(input1),
    .input2(input2),
    .input3(input3),

    .c00(c00),
    .c01(c01),
    .c10(c10),
    .c11(c11),

    .clear(clear),
    .a_data0(a_data0),
    .a_data1(a_data1),
    .b_data0(b_data0),
    .b_data1(b_data1),

    .done(done),
    .host_outdata(host_outdata)
  );

endmodule