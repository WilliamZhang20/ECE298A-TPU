`default_nettype none

module control_unit (
    input wire clk,
    input wire rst,
    input wire load_en,

    // Memory interface  
    output reg [2:0] mem_addr,

    // MMU feeding control
    output reg mmu_en,
    output reg [2:0] mmu_cycle,

    // For debugging
    output wire [1:0] state_out
);

    // STATES
    localparam [1:0] S_IDLE                  = 2'b00;
    localparam [1:0] S_LOAD_MATS             = 2'b01;
    localparam [1:0] S_MMU_FEED_COMPUTE_WB   = 2'b10;

    reg [1:0] state, next_state;
    reg [2:0] mat_elems_loaded;

    assign state_out = state;

    // Next state logic
    always @(*) begin
        next_state = state;

        case (state)
            S_IDLE: begin
                if (load_en) begin
                    next_state = S_LOAD_MATS;
                end
            end
            
            S_LOAD_MATS: begin
                // All 8 elements loaded (4 for each matrix)
                if (mat_elems_loaded == 3'b111) begin 
                    next_state = S_MMU_FEED_COMPUTE_WB;
                end
            end
            
            S_MMU_FEED_COMPUTE_WB:
                next_state = S_MMU_FEED_COMPUTE_WB;
               /* Cycle 0: Start feeding data (a00×b00 starts)
                * Cycle 1: First partial products computed, more data fed
                * Cycle 2: c00 ready (a00×b00 + a01×b10), can be output
                * Cycle 3: c01 and c10 ready simultaneously:
                *          c01 = a00×b01 + a01×b11
                *          c10 = a10×b00 + a11×b10
                * Cycle 4: c11 ready (a10×b01 + a11×b11), can be output
                * Cycle 5: All outputs remain valid
                */

			default: begin
				next_state = S_IDLE;
			end
        endcase
    end

    // State Machine
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE;
            mat_elems_loaded <= 0;
            mmu_cycle <= 0;
            mmu_en <= 0;
            mem_addr <= 0;
        end else begin
            state <= next_state;
            mem_addr <= 0;
            case (state)
                S_IDLE: begin
                    mat_elems_loaded <= 0;
                    mmu_cycle <= 0;
                    mmu_en <= 0;
                    if (load_en) begin
                        mat_elems_loaded <= mat_elems_loaded + 1;
                        mem_addr <= mat_elems_loaded + 1;
                    end
                end

                S_LOAD_MATS: begin
                    if (load_en) begin
                        mat_elems_loaded <= mat_elems_loaded + 1;
                        mem_addr <= mat_elems_loaded + 1;
                    end

                    if (mat_elems_loaded == 3'b101) begin
                        mmu_en <= 1;
                    end else if (mat_elems_loaded >= 3'b110) begin
                        mmu_en <= 1;
                        mmu_cycle <= mmu_cycle + 1;
                        if (mat_elems_loaded == 3'b111) begin 
                            mat_elems_loaded <= 0;
                            mem_addr <= 0;
                        end
                    end
                end

                S_MMU_FEED_COMPUTE_WB: begin
                    // Now: the TPU will be forever stuck in this cycle...
                    // Cycles through counter of 8...
                    // In each cycle of 8 counts, it will: output 4 16-bit output elements the result of the previous matmul,
                    // and take in 8 new 8-bit elements
                    if (load_en) begin
                        mat_elems_loaded <= mat_elems_loaded + 1;
                        mem_addr <= mat_elems_loaded + 1;
                    end
					mmu_cycle <= mmu_cycle + 1;
                    if (mmu_cycle == 3'b111) begin
                        mmu_cycle <= 0;
                    end else if (mmu_cycle == 1) begin
                        mat_elems_loaded <= 0;
                        mem_addr <= 0;
                    end
                end
				
				default: begin
					mat_elems_loaded <= 0;
                    mmu_cycle <= 0;
                    mmu_en <= 0;
				end
            endcase
        end
    end

endmodule
