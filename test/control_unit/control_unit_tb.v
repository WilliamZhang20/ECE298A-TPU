`default_nettype none
`timescale 1ns / 1ps

module control_unit_tb ();

    // Dump the signals to a VCD file. You can view it with gtkwave or surfer.
    initial begin
`ifdef VCD_PATH
        $dumpfile(`VCD_PATH);
`else
        $dumpfile("control_unit_tb.vcd");
`endif
        $dumpvars(0, control_unit_tb);
        #1;
    end

    // Inputs
    reg clk;
    reg rst;
    reg load_en;
    reg[1:0] state_out;

    // Outputs
    reg [2:0] mem_addr;
    reg mmu_en;
    reg [2:0] mmu_cycle;

    control_unit dut (
        .clk(clk),
		.rst(rst),
        .load_en(load_en),
        .mem_addr(mem_addr),
        .mmu_en(mmu_en),
        .mmu_cycle(mmu_cycle),
        .state_out(state_out)
    );

endmodule