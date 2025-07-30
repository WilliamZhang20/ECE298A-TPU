`default_nettype none
`timescale 1ns / 1ps

module memory_tb ();

	// Dump the signals to a VCD file. You can view it with gtkwave or surfer.
	initial begin
`ifdef VCD_PATH
		$dumpfile(`VCD_PATH);
`else
		$dumpfile("memory_tb.vcd");
`endif
		$dumpvars(0, memory_tb);
		#1;
	end

	// Inputs
	reg clk;
	reg rst;
	reg load_en;
	reg [2:0] addr;
	reg [7:0] in_data;

	// Outputs
	wire [7:0] weight0, weight1, weight2, weight3;
	wire [7:0] input0, input1, input2, input3;

	// Instantiate the memory module
	memory dut (
		.clk(clk),
		.rst(rst),
		.load_en(load_en),
		.addr(addr),
		.in_data(in_data),
		.weight0(weight0),
		.weight1(weight1),
		.weight2(weight2),
		.weight3(weight3),
		.input0(input0),
		.input1(input1),
		.input2(input2),
		.input3(input3)
	);

endmodule
