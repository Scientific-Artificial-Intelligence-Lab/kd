import numpy as np
import matplotlib.pyplot as plt


def plot_figures(context, config):

    from matplotlib.pyplot import MultipleLocator

    # 兼容 config 控制 use_metadata
    use_metadata = getattr(config, "use_metadata", False)
    n = context.n
    m = context.m
    n_origin = context.n_origin
    m_origin = context.m_origin
    u = context.u
    u_origin = context.u_origin
    ut = context.ut
    ux = context.ux
    uxx = context.uxx
    ut_origin = context.ut_origin
    ux_origin = context.ux_origin
    uxx_origin = context.uxx_origin

    right_side = context.right_side
    right_side_full = context.right_side_full
    right_side_origin = context.right_side_origin
    right_side_full_origin = context.right_side_full_origin


    # 数据切片索引，局部变量
    n1 = context.n1
    n2 = context.n2
    m1 = context.m1
    m2 = context.m2
    n1_origin = context.n1_origin
    n2_origin = context.n2_origin
    m1_origin = context.m1_origin
    m2_origin = context.m2_origin
    # Plot the flow field
    plt.figure(figsize=(10,3))
    mm1=plt.imshow(u, interpolation='nearest',  cmap='Blues', origin='lower', vmax=np.max(u_origin), vmin=np.min(u_origin))
    plt.colorbar().ax.tick_params(labelsize=16) 
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Metadata Field', fontsize = 15)
    plt.savefig(config.problem_name + '_Metadata_field_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    plt.figure(figsize=(10,3))
    mm1=plt.imshow(u_origin, interpolation='nearest',  cmap='Blues', origin='lower', vmax=np.max(u_origin), vmin=np.min(u_origin))
    plt.colorbar().ax.tick_params(labelsize=16) 
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Original Field', fontsize = 15)
    plt.savefig(config.problem_name + '_Original_field_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    # Plot the PDE terms
    fig=plt.figure(figsize=(5,3))
    ax = fig.add_subplot(1, 1, 1)
    x_index = np.linspace(0,256, n_origin)
    x_index_fine = np.linspace(0,100, n)
    if use_metadata == True:
        plt.plot(x_index_fine, ut[:,int(m/2)], color='red', label = 'Metadata')
    plt.plot(x_index, ut_origin[:,int(m_origin/2)], color='blue', linestyle='--') #, label = 'Raw data'
    ax.set_ylabel('$U_t$', fontsize=18)
    ax.set_xlabel('x', fontsize=18)
    x_major_locator=MultipleLocator(32)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.savefig(config.problem_name + '_Ut_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight') 

    fig=plt.figure(figsize=(5,3))
    ax = fig.add_subplot(1, 1, 1)
    x_index = np.linspace(0,256, n_origin)
    x_index_fine = np.linspace(0,100, n)
    if use_metadata == True:
        plt.plot(x_index_fine, ux[:,int(m/2)], color='red', label = 'Metadata')
    plt.plot(x_index, ux_origin[:,int(m_origin/2)], color='blue', linestyle='--') #, label = 'Raw data'
    ax.set_ylabel('$U_x$', fontsize=18)
    ax.set_xlabel('x', fontsize=18)
    x_major_locator=MultipleLocator(32)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.savefig(config.problem_name + '_Ux_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight') 

    fig=plt.figure(figsize=(5,3))
    ax = fig.add_subplot(1, 1, 1)
    x_index = np.linspace(0,100, n_origin)
    x_index_fine = np.linspace(0,100, n)
    if use_metadata == True:
        plt.plot(x_index_fine, uxx[:,int(m/2)], color='red', label = 'Metadata')
    plt.plot(x_index, uxx_origin[:,int(m_origin/2)], color='blue', linestyle='--', label = 'Raw data')
    ax.set_ylabel('$U_x$'+'$_x$', fontsize=18)
    ax.set_xlabel('x', fontsize=18)
    plt.legend(loc='upper left')
    plt.savefig(config.problem_name + '_Uxx_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight') 

    plt.figure(figsize=(5,3))
    x_index = np.linspace(0,100, n_origin)
    x_index_fine = np.linspace(0,100, n)
    if use_metadata == True:
        plt.plot(x_index_fine, u[:,int(m/2)], color='red', label = 'Metadata')
    plt.plot(x_index, u_origin[:,int(m_origin/2)], color='blue', linestyle='--', label = 'Raw data')
    plt.title('U')
    plt.legend(loc='upper left')
    plt.savefig(config.problem_name + '_U_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    plt.figure(figsize=(5,3))
    x_index = np.linspace(0,100, (n2_origin-n1_origin))
    x_index_fine = np.linspace(0,100, (n2-n1))
    if use_metadata == True:
        plt.plot(x_index_fine, right_side[:,int((m2-m1)/2)], color='red', label = 'Metadata')
    plt.plot(x_index, right_side_origin[:,int((m2_origin-m1_origin)/2)], color='blue', linestyle='--', label = 'Raw data')
    plt.title('Right side')
    plt.legend(loc='upper left')
    plt.savefig(config.problem_name + '_Right_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    plt.figure(figsize=(5,3))
    x_index = np.linspace(0,100, n_origin)
    x_index_fine = np.linspace(0,100, n)
    if use_metadata == True:
        plt.plot(x_index_fine, (ut-right_side_full)[:,int(m/2)], color='red', label = 'Metadata')
    plt.plot(x_index, (ut_origin-right_side_full_origin)[:,int(m_origin/2)], color='blue', linestyle='--', label = 'Raw data')
    plt.title('Residual')
    plt.legend(loc='upper left')
    plt.savefig(config.problem_name + '_Residual_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    plt.figure(figsize=(10,3))
    mm1=plt.imshow((ut-right_side_full), interpolation='nearest',  cmap='Blues', origin='lower', vmax=np.max((ut_origin-right_side_full_origin)), vmin=np.min((ut_origin-right_side_full_origin)))
    plt.colorbar().ax.tick_params(labelsize=16) 
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Metadata Residual', fontsize = 15)
    plt.savefig(config.problem_name + '_Metadata_Residual_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    plt.figure(figsize=(10,3))
    mm1=plt.imshow((ut_origin-right_side_full_origin), interpolation='nearest',  cmap='Blues', origin='lower', vmax=np.max((ut_origin-right_side_full_origin)), vmin=np.min((ut_origin-right_side_full_origin)))
    plt.colorbar().ax.tick_params(labelsize=16) 
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Original Residual', fontsize = 15)
    plt.savefig(config.problem_name + '_Original_Residual_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    plt.show()



def plot_metadata_diagnostics(context):
    """
    Plots diagnostic figures to compare original data with generated metadata.
    This logic is migrated from the old Data_generator.py.
    """
    print("INFO: Generating metadata diagnostic plots...")
    
    # Recreate necessary variables from context for comparison
    u_origin = context.u_origin
    u_meta = context.u # In this context, self.u holds the metadata
    n_origin, m_origin = context.n_origin, context.m_origin
    n_meta, m_meta = context.n, context.m

    # Downsample the high-resolution metadata for comparison
    u_meta_downsampled = np.zeros_like(u_origin)
    ratio = n_meta // n_origin
    for i in range(n_origin):
        for j in range(m_origin):
            u_meta_downsampled[i, j] = u_meta[i * ratio, j * ratio]

    # Calculate the difference
    # Adding a small epsilon to avoid division by zero
    diff = (u_origin - u_meta_downsampled) / (u_meta_downsampled + 1e-8)

    print("--- Metadata Generation Quality ---")
    print(f"Max difference: {np.max(diff):.4f}")
    print(f"Min difference: {np.min(diff):.4f}")
    print(f"Mean absolute difference: {np.mean(np.abs(diff)):.4f}")
    print(f"Median absolute difference: {np.median(np.abs(diff)):.4f}")
    print("------------------------------------")

    # Plot 1: Difference heatmap
    plt.figure(figsize=(10, 3))
    plt.imshow(np.abs(diff), interpolation='nearest', cmap='Blues', origin='lower', vmax=0.05, vmin=0)
    plt.colorbar().ax.tick_params(labelsize=16) 
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Metadata vs Original Data (Relative Difference)', fontsize=15)
    plt.savefig(f"{context.config.problem_name}_metadata_diff_heatmap.png", dpi=300, bbox_inches='tight')

    # Plot 2: Difference histogram
    plt.figure(figsize=(5, 3))
    plt.hist(diff.flatten(), bins=50)
    plt.title('Histogram of Relative Difference', fontsize=15)
    plt.savefig(f"{context.config.problem_name}_metadata_diff_hist.png", dpi=300, bbox_inches='tight')

    # Plot 3: Slice comparison
    plt.figure(figsize=(5, 3))
    x_index_origin = np.linspace(0, 100, n_origin)
    x_index_meta = np.linspace(0, 100, n_meta)
    time_slice_idx_origin = m_origin // 2
    time_slice_idx_meta = m_meta // 2
    plt.plot(x_index_origin, u_origin[:, time_slice_idx_origin], label='Original Data')
    plt.plot(x_index_meta, u_meta[:, time_slice_idx_meta], linestyle='--', label='Generated Metadata')
    plt.title('Data Slice Comparison', fontsize=15)
    plt.legend()
    plt.savefig(f"{context.config.problem_name}_metadata_slice_comparison.png", dpi=300, bbox_inches='tight')
    
    plt.show() # Optional: to show plots immediately