`default_nettype none

module control_unit (
    input wire clk,
    input wire rst,
    input wire instrn,

    // Memory interface  
    output reg mem_load_mat,
    output reg [2:0] mem_addr, // Adjusted to 3 bits for matrix and element selection

    // MMU feeding control
    output reg mmu_en,
    output reg [2:0] mmu_cycle // Renamed from mmu_cycle, adjusted to 3 bits
);

    // Instruction decoding
    wire load_en = instrn;          // Bit 0: load enable
    // Bit 7: ignored as specified

    // STATES
    localparam [1:0] S_IDLE                  = 2'b00;
    localparam [1:0] S_LOAD_MATS             = 2'b01;
    localparam [1:0] S_MMU_FEED_COMPUTE_WB   = 2'b10;

    reg [1:0] state, next_state;
    reg [2:0] mat_elems_loaded;

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
            
            S_MMU_FEED_COMPUTE_WB: begin
               /* Cycle 0: Start feeding data (a00×b00 starts)
                * Cycle 1: First partial products computed, more data fed
                * Cycle 2: c00 ready (a00×b00 + a01×b10), can be output
                * Cycle 3: c01 and c10 ready simultaneously:
                *          c01 = a00×b01 + a01×b11
                *          c10 = a10×b00 + a11×b10
                * Cycle 4: c11 ready (a10×b01 + a11×b11), can be output
                * Cycle 5: All outputs remain valid
                */
                if (mmu_cycle == 3'b101) begin
                    next_state = S_IDLE;
                end
            end

			default begin
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
            mem_load_mat <= 0;
            mem_addr <= 0;
        end else begin
            state <= next_state;
            case (state)
                S_IDLE: begin
                    mat_elems_loaded <= 0;
                    mmu_cycle <= 0;
                    mmu_en <= 0;
                    mem_load_mat <= load_en;
                    mem_addr <= 0;
                end

                S_LOAD_MATS: begin
                    if (load_en) begin
                        mat_elems_loaded <= mat_elems_loaded + 1;
                        mem_load_mat <= 1; // enable writes into memory
                        mem_addr <= mat_elems_loaded + 1; // Decode instruction
                    end else begin
                        mem_load_mat <= 0;
                        mem_addr <= 0;
                    end

                    if (mat_elems_loaded == 3'b111) begin 
                        mat_elems_loaded <= 0;
						mmu_en <= 1;
					end
                end

                S_MMU_FEED_COMPUTE_WB: begin
                    mmu_en <= 1;
                    mem_load_mat <= 0;
                    mem_addr <= 0;
					mmu_cycle <= mmu_cycle + 1;
                end
				
				default begin
					mat_elems_loaded <= 0;
                    mmu_cycle <= 0;
                    mmu_en <= 0;
                    mem_load_mat <= load_en;
                    if (load_en) begin
                        mem_addr <= 0;
                    end else begin
                        mem_addr <= 0;
                    end
				end
            endcase
        end
    end

endmodule
